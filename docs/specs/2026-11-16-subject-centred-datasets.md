# Subject-centred datasets: the subject as the unit, the dataset as a catalogue, templates as shared subjects (spec, 2026-11-16)

Status: **design, no code.** The maintainers approved the direction on 2026-11-16 (a subject-centred design) and decided its open questions the same
day; each decision is marked "decided 2026-11-16" where it applies, and section 10 lists them. This spec says what the format becomes, what replaces
the staging and merge of `docs/specs/2026-11-14-dataset-wide-runs.md` section 2 (designed, never built), how existing datasets move, and the
technical questions still open (section 10).

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

**Two modules, kept apart.** The design has two parts, each with its own files and its own code in io:
- **The subject module** (section 2, 4): a subject package, self-contained. Opening, reading, computing on and writing ONE subject needs that
  package and nothing else: no catalogue, no participants table, no index, no other subject's file. A template is opened exactly as any subject is.
- **The dataset module** (section 3, 5, 6): the catalogue, the participants table, cohorts and filters, the cross-subject index, group results.
  It reads subject packages; a subject package never reads it. It is needed only for work across subjects and for browsing a dataset.

The dependency goes one way (the dataset module depends on the subject module, never the reverse), as the library dependency rule does for io and
compute. This is also how the Studio already works: it opens one subject at a time, and its dataset pages are separate screens (section 7).

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
  - its **home** `dataset` row (the dataset it was created in: provenance, not membership; section 3). The model's `datastore` table stays in the
    schema for now and holds one row recording where the package was made, as a hint only: nothing reads it to find data (the datastore is
    obsolete, section 3);
  - copies of the **shared vocabulary** rows its rows name: app- and dataset-scope dictionaries with their entries (atlas levels, `frequency bands`,
    `recipes`), as nxr's fragment carries "the shared rows";
  - **anchor rows** of the template it is registered to (section 4.3): the few rows of another package its own rows point at.
- **`small_files(key, value)`**, format 52's table, unchanged: every `zarr.json` and every value under 1 MiB (`kSmallFileThreshold`).
- **`meta`**: `schema_version` (the model's `PRAGMA user_version`), `store_format` = 53, `built_by`, and the package's own record (section 2.3).
- **`key_digest(key, size_bytes, sha256)`** (new): the size and SHA-256 of every key's value, written by the store when the value is written
  (`set`, and a streamed writer's `close`, which hashes as it streams). It is what the content hash (section 2.3) and the migration's verification
  are computed from without re-reading the shards.

**Why one file for rows and small keys** (technical question 1): the rows and the keys a node writes are then in ONE SQLite database, so a node's rows
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
  alternative: technical question 3.)

### 2.2 The schema version, per subject

- A package carries its own versions: `PRAGMA user_version` (the model's schema version) and `meta.store_format` (53).
- **A reader refuses a package at a version it does not know**, by name, as today a reader refuses a dataset database (`schema_version` error). It
  never opens it partly.
- The model's rule that one database is at one version, entirely, becomes: **a reader works on packages at versions it knows; an operation across
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

A dataset is the dataset module's files: a **catalogue**, optionally a **participants table** and an **index**, and group packages, in one folder
that lives anywhere and is addressed by its own path (`--dataset PATH`).

**The datastore is obsolete** (decided 2026-11-16). Datasets no longer share a parent folder, and nothing requires one: two datasets on two disks,
or a dataset inside a project folder, are equally datasets. The datastore's `catalog.json` (its list of datasets) retires with it: no build reads or
writes it. A list of recently opened datasets is a per-user convenience of an application (section 7.1), never part of the format.

**What a dataset is** (decided 2026-11-16): a folder holding its catalogue and, by default, its subjects' packages, so it is copied and archived as
one. A catalogue may also list packages outside its folder, by path, read-only to it (section 3.2).

```
<dataset>/                            any path
  catalogue.sqlite                    the dataset: its subjects and its template references
  participants.sqlite                 the participants table (section 3.1); optional, its own file so it can be withheld
  .writer.lock                        held only while the dataset module's files are written (registration, participants), seconds
  index.sqlite                        a CACHE (section 3.3); deletable
  <group>.nxr.zarr/                   group packages (section 5)
  <subject>.nxr.zarr/                 subject packages (the subject module)
```

### 3.1 What the catalogue holds

The catalogue is a SQLite file whose tables are, where the model already has them, **the model's own** (`model.sql`), so the three
implementations read it with code they already have:

| table | from | holds |
|---|---|---|
| `meta` | model | `catalogue_version`, `built_by` |
| `dataset` | model | the dataset's own row (id, name, source); its `path` is the folder's name, the dataset is found by the path the caller gives |
| `subject` | model | one row per member: id, name, `path` (relative to the dataset folder, or a location outside it: section 3.2), kind (`subject`, `template`, `group`), `template_id`, provenance columns. A COPY of the package's own row, refreshed at registration |
| `subject_entry` | new | per member: `subject_id`, `schema_version`, `store_format`, `content_hash`, `registered_utc`, `registered_by`, `status` (`registered`, `missing`, `stale`, `replaced`), and the package's counts taken at registration (`n_acquisitions`, `n_recordings`, `n_surfaces`, `n_fields`, `n_operators`: the columns of the model's `subject_status` view, so a dataset's subject list needs no package opened) |
| `template_reference` | new | the templates the dataset's subjects are registered to: template name, version, the template package's subject id, content hash, a location hint (section 4) |

The catalogue holds **no** manifold, field, selection, operator or measurement row, and no participant variable. It is small (kilobytes for hundreds
of subjects). A job on one subject never needs it; a job across subjects reads it to find its inputs.

**The participants table is its own file**, `participants.sqlite`, holding the model's `variable_dictionary` (what is recorded about the subjects:
name, type, unit, levels; BIDS `participants.json`) and `subject_variable` (each subject's values; BIDS `participants.tsv`). The model has had these
tables since its first version; nothing wrote them. It is separate because participant variables (age, sex, diagnosis, group) can be the most
sensitive part of a dataset: a dataset can be shared, published or copied to a cluster with its catalogue and packages and WITHOUT its participants
table, and the subjects and the catalogue never need it. Which variables it may hold, where they come from and when it may travel with a dataset are
decided 2026-11-16: the table is its own file, never copied, exported or published unless the command names it (`--with-participants`), and never
needed to open or compute on a subject. Its variables come from BIDS `participants.tsv` (and its JSON sidecar) or a cohort's own records, imported
by `participants import`.

**Participants**: `participants import --dataset PATH --from participants.tsv [--sidecar participants.json]` writes the variables (BIDS's
`participant_id` column matched to subject names; an unknown participant is refused by name, a subject without a row has no values: a missing value
is a missing row, as the model says); `participants export` writes the TSV and JSON back. Values are typed (`number`, `string`, `boolean`) as the
model's `variable_dictionary.type` says.

#### Selecting subjects: `--subjects` and `--where` (decided 2026-11-16)

Every command that works across subjects takes **exactly one** of:
- `--subjects NAME,NAME,...`: an explicit list of subject names (or ids) of the catalogue. An unknown name is refused, by name.
- `--where FILTER`: a filter over the **participant variables only**, in the small language below. It is parsed and type-checked by nxr-neuro's io
  and evaluated over the participants table; it is never passed to SQL as text, and it cannot reach any other table.

Giving neither is refused; there is no "all subjects" default and no `all` keyword. Giving both is refused. A selection that matches no subject is
refused. Before computing, the command prints the matched subjects (name, id, package content hash) and their number; the same list, the filter
text and the filter language's version go into the result's provenance.

**The filter language, version 1** (EBNF; keywords are lower-case words; whitespace separates tokens):

```
filter      = disjunction ;
disjunction = conjunction , { "or" , conjunction } ;
conjunction = negation , { "and" , negation } ;
negation    = "not" , negation | primary ;
primary     = "(" , disjunction , ")" | test ;
test        = variable , comparator , value
            | variable , "in" , "(" , value , { "," , value } , ")"
            | variable , "is" , "missing" ;
comparator  = "==" | "!=" | "<" | "<=" | ">" | ">=" ;
variable    = name | "`" , { any character except "`" } , "`" ;     (* a name with spaces or symbols is quoted with backticks *)
name        = letter , { letter | digit | "_" } ;
value       = number | string | "true" | "false" ;
number      = [ "-" ] , digit , { digit } , [ "." , digit , { digit } ] , [ ( "e" | "E" ) , [ "+" | "-" ] , digit , { digit } ] ;
string      = "'" , { any character except "'" | "''" } , "'" ;        (* '' is a quote inside a string *)
```

Precedence: `not` binds tightest, then `and`, then `or`; parentheses group. Examples: `group == 'control' and age >= 60`,
`site in ('a', 'b') and not (diagnosis is missing)`, `` `handedness score` > 40 or group != 'patient' ``.

**Checks, before anything is read** (each refusal names what is wrong):
- **Unknown variable**: refused, naming the variables the participants table has. No participants table: refused, saying so.
- **Types**, from `variable_dictionary.type`: a `number` variable takes numbers and every comparator; a `string` variable takes strings with `==`,
  `!=` and `in` (no ordering of text); a `boolean` variable takes `true` / `false` with `==` and `!=`. A mismatch is refused (e.g. `age == 'old'`).
- **Levels**: a categorical variable (`levels_json`) refuses a value outside its levels, naming them.
- **Syntax**: an error names its position and what was expected (`=` alone is refused with a hint to write `==`).

**Missing values** follow SQL's three-valued logic, so a filter never selects a subject by accident: a comparison or `in` on a missing value is
*unknown*; `not unknown` is unknown; `unknown and false` is false, `unknown or true` is true; a subject is selected only when the whole filter is
TRUE. `is missing` is the one test that is true on a missing value.

**Expandable later**: a new construct (functions, ranges, variables of another table) is a new language version, recorded in provenance; a
version-1 filter keeps its meaning.

### 3.2 Membership and location

- A subject is a member of a dataset when the catalogue has its entry. Its package is usually inside the dataset folder; a package elsewhere
  (another dataset's, a shared read-only location) can be listed by path. **A catalogue writes only packages inside its own folder**; one it lists from
  elsewhere is read-only to it (decided 2026-11-16, as is the next point).
- The package's `dataset` row is its **home** (where it was made). Listing a subject in another catalogue does not change the package.
- **A subject's UUID is global**: one subject, one id, wherever its package is copied. A copy that is then changed is a new content hash under the
  same id (registration reports it `updated`); a deliberate fork takes a new id (`subject fork`).

### 3.3 The cross-subject index is a cache

Some readers want SQL across subjects (the Studio's views over a dataset, `measure query` across a cohort). For them `dataset index --dataset
PATH` builds `index.sqlite`: a database in the model's schema holding every registered package's rows and its templates' rows, in the
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
**A template is opened exactly as any subject is**: the same package, the same bundle, the same readers; only its kind and its release record
differ.

### 4.2 Shared and versioned

- **A template is released**: `template release --package PATH --name NAME --version VERSION` freezes a template package: it computes the content
  hash, writes `name`, `version` and the hash into the package record, and from then on io opens it **read-only** (a released template is never
  written again; a change is a new version, a new package with a new subject id).
- **Released versions are frozen** (decided 2026-11-16). `template release` is the only way to publish a version; it records who released it (the
  program and the releaser's stated name) and refuses an item without an `origin` record (source, licence, attribution), so a template's licences
  are checked before it is shared. The version string is the releaser's (e.g. `1.0`); `name@version` must be unique, and its content hash decides.
- **Where templates live: anywhere, found through the template search path** (decided 2026-11-16). A dataset never copies a template: its
  catalogue's `template_reference` (and each package's record) names it by **name + version + content hash**. io finds it by looking, in order, in:
  1. the locations the command names (`--templates PATH`, repeatable);
  2. the environment variable **`NXR_TEMPLATE_PATH`**: a list of folders separated by the platform's path separator (`:` on Linux and macOS, `;` on
     Windows), as `PATH` is;
  3. the user setting **`template_path`** (a JSON list of folders) in nxr's per-user settings file, `$XDG_CONFIG_HOME/nxr/settings.json`
     (`~/.config/nxr/settings.json` when unset) on Linux, `~/Library/Application Support/nxr/settings.json` on macOS,
     `%APPDATA%\nxr\settings.json` on Windows; the Studio's settings page edits the same file;
  4. the location hint recorded in the catalogue entry (where the template was when the dataset was made).

  In each folder a released template is the package **`<name>@<version>.nxr.zarr`**. The first one found whose content hash equals the reference's
  is used; one with the right name and version but another hash is skipped and reported (both hashes). None found: refused, listing every location
  searched. The search path says only WHERE to look; WHICH template is used is fixed by the reference (name, version, hash), so the setting never
  chooses data. `template list` prints every template on the search path with its version and hash.
- **Which template a dataset uses** is the template its data were computed in, named by the caller at `dataset create --template NAME@VERSION` (or by
  the importer, section 4.5) and recorded in the catalogue; never a default of the code.
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
  checks every anchor row against the template's own. Because the anchor rows are in the package, **a subject's bundle is complete without opening
  its template**; the template package is opened only when a template's ARRAYS are read (to draw its pial, to read its tiles). (Alternatives: technical
  question 2.)
- **The registration itself stays nodes of the subject**, as today: its registered sphere (`<surface>_sphere`), its group tree per subject
  (`<surface>_group_tiles`, with `sphere_match_median` and `sphere_match_max`), and anything carried to the template (resampled scalars, rotated
  vectors). Their `producer_json` names the template by name, version and hash.
- **Changing template version** is a re-registration: the subject's carried nodes are recomputed against the new version (new nodes, new cache keys;
  the template version is part of every such node's cache key). Old nodes keep naming the old version until removed. A released version is never
  changed, so a subject is never invalidated by a template under it; it is moved to a new version only when the caller asks (`subject retemplate
  --package PATH --template NAME@VERSION`; decided 2026-11-16). A cohort is never pooled across template versions.

### 4.4 Several templates, two editions, and registrations between templates (decided 2026-11-16)

- **More than one template.** fsaverage (at its resolutions) is one template; Brainstorm's default anatomy **ICBM152 (2023b)**, FreeSurfer-derived
  surfaces with their registration spheres, is another template package of its own; others can follow. Each is built by `template create` from its
  sources, with an `origin` record on every item, and released by name and version.
- **Two editions of one template.** A template may exist as a **full** edition (every item, for use where its terms allow) and a
  **redistributable** edition (only the items whose `origin` records allow redistribution). The edition is part of the package record and of every
  reference: `NAME@VERSION` names the full edition, `NAME@VERSION:redistributable` the other; each has its own content hash. `template release
  --edition redistributable` makes the second from the first: it keeps an item only when its `origin` licence allows redistribution, refuses an item
  with no `origin` record, and lists what it left out. Items kept are byte for byte the full edition's (their `key_digest` entries are equal), so an
  analysis that reads only kept items gives the same result on either edition, and that is checkable.
- **Registrations between templates are their own versioned packages.** A registration from one template to another (e.g. ICBM152's sphere to
  fsaverage's sphere) is a package of the new kind **`registration`** (schema 51 adds the kind), released and versioned like a template, found on
  the template search path as `<name>@<version>.nxr.zarr`. It holds:
  - the resampling operators between the two templates' surfaces (the model's `operator` rows, `from_manifold_id` on one template and
    `to_manifold_id` on the other, reached through anchor rows of both; the arrays a sparse matrix, as the model stores operators), in each
    direction it supports;
  - its record: both templates by name, edition, version and hash; the method (e.g. nearest vertex or barycentric weights on the registered
    spheres); its `origin`; and what it cannot resolve (`producer_json.warnings`: the correspondence is the spheres', not anatomy's; values are
    smoothed by resampling).
  `registration create --from NAME@VERSION --to NAME@VERSION --method METHOD` builds one (the operator is computed by compute's functions,
  composed by the pipelines, stored by io), and `registration release` freezes it.
- **Datasets in different template spaces are compared by resampling RESULTS through a registration, never by converting a dataset.** A dataset's
  kernels, source estimates and results stay on the surfaces they were computed on. Only what a group analysis reads (section 5) is resampled, at
  the moment it is read, and the resampled values carry the registration in their provenance.

### 4.5 The importer rule (decided 2026-11-16)

- **A subject without its own MRI**, whose data were source-mapped on a default anatomy (Brainstorm's "use default anatomy"), gets that anatomy as
  **its own anatomy**: the importer writes the anatomy's surfaces (and spheres) into the subject's package as the subject's own nodes, each with the
  `origin` record of the template item it came from, so its kernels and results sit on surfaces of the subject, exactly as they were computed. Its
  registration to that template is the identity, recorded as such.
- **The dataset's template is the template package its data were computed in**, recorded by name, edition and version. The importer finds it by
  comparing the source's anatomy with the released templates on the search path (surfaces byte for byte, or by their digests); it never records
  "whatever the source protocol had", and it never falls back to a default. When no released template matches, the import is refused, naming what
  was found, until the caller names one (`--template NAME@VERSION`) or a matching template is created and released.

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
4. **Where group results live: a group package** (decided 2026-11-16). Today `pyramid reduce` writes into the default subject's store. A
   released template is read-only, and a group result is not part of an anatomy, so the result goes to a package of kind `group` (the model's own
   kind for "group averages and statistics": Brainstorm's `Group_analysis`) inside the dataset, registered in the catalogue like any member:
   - its fields are the same rows `pyramid reduce` writes today (`group <kind>`, `function = sum`, `field_contribution` rows, measurements on the
     template's tile partition, which the group package reaches through anchor rows as a subject does);
   - `field_contribution.subject_id` and `source_field_id` name a member's rows, which live in the member's package. As a subject does with its
     template, the group package carries them as **anchor rows**: each member's `subject` row and the source field rows its contributions name
     (what `bundleOf` returns as `related_subjects` and `lineage_fields`), with the member's content hash in `producer_json`. No member array is
     copied;
   - one group package per analysis the caller names (`--into GROUP_NAME`), never a default one; a re-run replaces its group fields as today.
   - The group statistics (the group map, sign-flip and random-rotation nulls, covariates from the participants table) are statistic nodes of the
     same group package.
5. **Across template spaces** (decided 2026-11-16): a group analysis whose members are registered to different templates (from one dataset or from
   several) must name **the target template** (`--target-template NAME@VERSION`) and **the registration** for every other template space it
   reads (`--registration NAME@VERSION`, repeatable; one per source template). Each member's inputs are resampled into the target space through
   that registration as they are read; the members' own packages are not changed. Without both, members in different spaces are refused, naming
   their templates. The target template and every registration used (name, edition, version, hash) are recorded with the result.

## 6. Writing on clusters

**One job writes one subject package, with no shared lock.**

- **The job** opens the subject's package (`--package PATH`; `--dataset PATH --subject NAME` is a convenience that looks the path up in the
  catalogue, read-only), takes **the package's** writer lock (`<subject>.nxr.zarr/.writer.lock`, the existing lock file and its rules: same-host
  stale detection, the handshake), and writes rows and keys into `subject.sqlite` and its shards. It needs no catalogue, reads its template only when
  a template's arrays are inputs, and never writes either. Two jobs on two subjects share no file they write.
- **Where `subject.sqlite` is written on a cluster** is decided by the canary (dataset-wide-runs section 0): in place (one process, one host: SQLite's
  WAL is safe when every connection is on one host), or on the node's local disk and moved into the package when the job closes (one rename on the
  same file system, else copy, verify the digest, then replace). Either way the shards go straight to the package.
- **Creating a subject** on a cluster: `subject create --package PATH --template NAME@VERSION` makes a package without touching any catalogue.
- **Registering** replaces staging and merge: `dataset register --dataset PATH --package PATH [PATH ...]` (or `--scan`, every package in
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
  local disk, or none (on a cluster the scheduler already places one job per allocation). It is never a shared file on shared storage (technical
  question 5).

What becomes of the dataset-wide-runs spec: section 1 (few files, format 52) stands and is extended here; **section 2 (staging and merge) is
superseded** (no merge of rows is needed when no shared database exists); section 3 (the group scope) is kept with section 5 above.

## 7. Reading

The Studio's `bundleOf` (`backend/src/db/bundle.ts`), nxr's reader (`Dataset.bundle`) and nxr-neuro's `subject_bundle` (the reference's port, used by
the WebAssembly data client) issue the same SQL for a subject. **The SQL does not change.** What changes is the database it runs on: the subject's
`subject.sqlite` instead of the dataset's `dataset.sqlite`. Because a package holds the anchor rows of its template (section 4.3), the
"following the template" part of the bundle (`related_subjects`, the template's manifolds and their parts, `lineage_fields`, `lineage_operators`) is
answered from the package alone; no ATTACH and no second database is needed to build a bundle.

### 7.1 The Studio: what actually changes (checked against its `main`)

The Studio already works one subject at a time: the Studio screen opens one subject from its bundle, and the dataset's pages (the catalog of
screens in `app/src/pages/catalog.ts`: Datasets, Subjects, Acquisitions, Recordings, Fields, …) are separate screens over the database's views. So
the change is WHERE it reads, not how it works:

| piece (Studio `main`) | today | in the new layout |
|---|---|---|
| `bundleOf` / `bundleByPath` (`backend/src/db/bundle.ts`) | SQL over a `BundleSource` (the dataset's database) | **unchanged**; given a `BundleSource` over the subject's `subject.sqlite` |
| `Model.fromBundle`, the entities, the Studio screen | built from the bundle | **unchanged** |
| the model service (`backend/src/service/model.ts`) | `sets`: one open `ModelDatabase` per dataset; `bundle(storePath)` picks the dataset's database by the path's first segment; `owner` / `forSubject` find the database that holds a row | one `ModelDatabase` per OPEN SUBJECT package, opened on demand; `bundle(storePath)` opens the package the path names; `owner` / `forSubject` resolve to that package's database (a row belongs to exactly one package) |
| the writer lock (`backend/src/service/writer-lock.ts`) | one per dataset | the same file, one per package (a subject's edits lock that subject only) |
| choosing where data is (`app/src/platform/electron.ts`; Home and the File menu: "Change datastore") | a datastore folder, holding datasets, configured once | **"Open dataset"**: the user picks a dataset folder at any path; the app keeps a list of recently opened datasets in its per-user settings (a convenience, not part of the format) |
| `writeCatalog()` after a write | rewrites the datastore's `catalog.json` from every dataset database | registers the changed package in its dataset's catalogue (section 6); `catalog.json` is retired |
| the default subject | a subject of the dataset's database | a template package, opened by the same `bundle(storePath)`; its arrays read from its own store (`Model.readFor` resolves the template's package from the reference instead of `<dataset>/<template path>`) |
| the Datasets and Subjects pages (`dataset`, `subject_status` views) | views over each dataset database | read from the catalogue: its `dataset` and `subject` rows, and `subject_entry`'s counts (the `subject_status` columns) |
| the cross-subject pages (Acquisitions, Recordings, Fields, Manifolds, Selections, Operators, …: `*_view`) | views over a dataset's database, every subject at once | the same views, over the open subject's package, or over `index.sqlite` (section 3.3) when a page asks for the whole dataset |
| the served build (`backend/src/served.ts`, `app/src/services/servedDatabase.ts`) | fetches the root's `catalog.json` and each dataset's whole `dataset.sqlite`, then runs `bundleByPath` | is given a published dataset's URL, fetches its `catalogue.sqlite`, and ONE `subject.sqlite` when a subject is opened (smaller downloads); the same `bundleByPath` |
| imports and conversions (`backend/src/db/import.ts`) | write into the dataset's database | write a package (section 6: `subject create --package`), then register it |

Nothing in the Studio's renderer, its engine or its GPU chains changes. The parity scripts (bundle, views, writes) run against packages.

### 7.2 nxr-neuro, nxr and the WebAssembly client

- **nxr-neuro**: `subject_bundle(db, subject_id)` runs unchanged on a package's database; io gains `SubjectPackage::open(path, read_only)` (the
  subject module) and `Catalogue::open(dataset_folder)` (the dataset module), in separate headers; the catalogue code includes the package code,
  never the reverse.
- **nxr**: `nxr.io.reader` opens a package directly (`open_subject(path)`), and a catalogue (`open_dataset(path)`, giving members' packages);
  `nxr.io.small_files` already reads the `small_files` table and now opens it in `subject.sqlite`.
- **The WebAssembly data client** (`wasmClient`): `bundle(storePath)` opens the package; `read`, `create`, `put`, `remove` act on its store;
  `status()` reads the open dataset's catalogue (a dataset is opened by its path or URL); `view(name, …)` runs on the open package or the index.
- **Arrays**: a member's keys are read from its package (`SmallFileStore` over the package folder, as format 52); a template's from the template
  package.
- **Parity**: the bundle from a package equals, row for row, the bundle of the same subject from a format-52 dataset database (synthetic datasets,
  the WebAssembly parity script's checks extended to packages; the same test in nxr against the Studio).

## 8. Migration

Existing datasets (one `dataset.sqlite`, store format 50, 51 or 52) move to the new layout with **`dataset split --dataset PATH`**. A datastore
folder's datasets are each split into a self-contained dataset **where they are**: nothing has to be moved, and the old `catalog.json` is left as it
was and no longer read. Each split runs in
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
   moved into a group package (decided 2026-11-16), verified as above.
7. **All at once, not on first use** (decided 2026-11-16): a dataset is split by one explicit command, never silently when a newer build
   opens it, so no reader meets a dataset half in one layout without being told. A build that knows format 53 still reads 50–52 as today.
8. **Reverse**: `dataset join --dataset PATH` writes a format-52 `dataset.sqlite` from the packages (the index build, written as the
   dataset database) and removes the packages' row tables, leaving `small_files` as format 52 had it. `split` then `join` gives back the original
   database, compared table by table (`test_split`: synthetic datasets at 50, 51 and 52, a crash at each stage, idempotence, the round trip).

**The version step**: store format **53** (packages and catalogue) and the model's schema version **51** (the new tables: `subject_entry`,
`template_reference`, `key_digest`, the package record in `meta`, the subject kind `registration`), added to `model.sql` in all three implementations at once. Every reader refuses a
version above the newest it knows (`schema_version` error), so a build from before this change refuses a format-53 package instead of misreading it,
and a format-53 build still opens 50–52 datasets as today.

**Order across the implementations** (decided 2026-11-16): nxr-neuro's io and its WebAssembly client first, behind the version step; then nxr; then
the Studio; each with parity on the same synthetic datasets before any real dataset is split.

**Stability** (decided 2026-11-16): the subject package format (store format 53, schema 51) is declared STABLE, with a promise that later builds keep
reading it, only after phase G and its three-way parity tests; until then it is experimental and may change.

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

## 10. Phases, decisions and the technical questions still open

**Phases** (each its own PR, with its tests; nothing on a real dataset without the maintainers' OK):

| phase | what | tests |
|---|---|---|
| 0 | the canary measurement (dataset-wide-runs section 0), extended: in-place vs node-local `subject.sqlite` | timings per phase |
| A | the subject module in io (`SubjectPackage`, `subject_bundle` over a package), reading packages made by a test from a format-52 dataset | bundle from package = bundle from the database |
| B | format 53 writing: `SubjectWriter` on a package (per-package lock), `key_digest`, the content hash, `subject create --package`, `subject verify` | node procedure, crash recovery, hash stable and sensitive |
| C | the catalogue: `dataset create` (catalogue), `dataset register`, `participants import/export`, `--subjects` and `--where` (the filter language, version 1), `dataset index` | conflict rules, idempotence, id and vocabulary clashes |
| D | templates: `template create`, `template release` (both editions), `template list`, the template search path, resolution by name + edition + version + hash, anchor rows; `registration create` and `release`; the importer rule | refusal of a wrong hash, anchor checks, the redistributable edition's items equal the full one's, a registration's round trip on synthetic spheres |
| E | migration: `dataset split`, `dataset join`, `dataset retemplate` | round trip at 50/51/52, crash at each stage |
| F | group: `pyramid reduce` over a catalogue cohort into a group package; `--target-template` and `--registration` across template spaces | the existing reduce tests and nxr parity, unchanged sums; refusal without a registration; resampled inputs equal the operator applied by hand |
| G | nxr and the Studio: the same source, catalogue and package writer, parity tests on the same synthetic datasets; the WebAssembly client | three-way parity |

**Decided 2026-11-16** (each is written into the sections above):

- **What a dataset is** (3): a folder of subject packages with its catalogue, by default; a catalogue may list packages outside it by path, read-only.
- **The datastore is obsolete** (3): datasets live anywhere and are addressed by their own path; `catalog.json` retires; recent datasets are an
  application's per-user list. Existing datastore folders' datasets are split in place (8).
- **Shared templates** (4.2, 4.3): released versions are frozen, published only by `template release` (which requires an `origin` record on every
  item), found anywhere by name + version + hash through the template search path (`--templates`, `NXR_TEMPLATE_PATH`, the user setting
  `template_path`, the recorded hint); a subject moves to a new version only when the caller asks (`subject retemplate`); a cohort is never pooled
  across versions.
- **Group results** (5): in a group package of the dataset; existing group fields in default subjects move into one at migration.
- **The participants table** (3.1): its own file, never copied, exported or published unless named, never needed for one subject.
- **Selecting subjects** (3.1): `--subjects` (an explicit list) or `--where` (the filter language, version 1, over participant variables only,
  parsed and checked by io, never SQL); neither is refused; the matched subjects and hashes are printed and recorded.
- **Subject identity** (3.2): a subject's UUID is global; one subject may be a member of several datasets, read-only in all but the one whose folder
  holds it.
- **Migration** (8): all at once by `dataset split`, not on first use; nxr-neuro first, then nxr, then the Studio, each with parity first.
- **Stability** (8): the format is declared stable after phase G's three-way parity, not before.
- **Several templates** (4.4): fsaverage and ICBM152 (2023b) are template packages, each possibly in a full and a redistributable edition (the
  `origin` records decide what the second keeps), both named and versioned.
- **Registrations between templates** (4.4): their own versioned packages (kind `registration`); datasets in different template spaces are compared
  by resampling results through them, never by converting a dataset.
- **The importer rule** (4.5): a subject without its own MRI gets the default anatomy it was mapped on as its own; the dataset's template is the
  template its data were computed in, recorded by name and version, never inferred from the source protocol, never a silent default.
- **Group analyses across template spaces** (5): name the target template and the registration; both are recorded with the result.

**Technical questions still open** (each with its recommendation):

1. **One file or two per subject**: rows and small keys in one `subject.sqlite` (recommended: one transaction for a node's rows and keys, one file
   fewer), or rows in their own file beside format 52's `small_files.sqlite` (no change to the small-key file's name).
2. **References into another package**: anchor rows copied into each package (recommended: the model's foreign keys stay enforced, a bundle needs
   one file, the rows are what `bundleOf` already follows), or foreign keys relaxed for columns that may point into another package (a change to
   `model.sql`), or the template's whole row set copied into each package (nxr's fragment does this; it duplicates rows, though not arrays).
3. **Ids of shared vocabulary rows**: per-copy ids matched by key at index time (recommended: nxr's merge, ported), or deterministic ids (a hash of
   scope, name and entries) so every copy has the same id.
4. **The catalogue's form**: SQLite with the model's tables (recommended: every implementation already reads them), or text files (a BIDS-like TSV
   and JSON pair, easy to diff and edit by hand). Participants import and export TSV either way.
5. **The chunk-queue lock**: per machine, named by the caller (recommended), or per package.
6. **The content hash**: digests written at every key write (recommended: no re-read of shards to verify), or computed on demand by `subject
   verify` only (no write cost, slow to check).

## Checklist (the architecture policy)

- Compute is untouched: packages, the catalogue, the index, registration, templates and the migration are io; the cohort and the group package are
  named by the caller and composed by the pipelines.
- Two modules in io, one dependency direction: the subject module never includes the dataset module; one subject is opened and computed on with its
  package alone.
- Every file access stays in io: a pipeline names a subject, a cohort or a template; io resolves and opens them.
- The caller names the data: the cohort (`--subjects` or `--where`; neither is refused), the template (name and version), the group package, the
  packages to register.
- Ported, not redesigned: the row set of a subject and the vocabulary remap (nxr's `shared_dataset` fragment and merge), the bundle's queries and its
  "following the template" (the Studio's `bundleOf`), the participants tables (the model's `variable_dictionary` and `subject_variable`), the writer
  lock, `dataset compact`'s verify-then-commit.
