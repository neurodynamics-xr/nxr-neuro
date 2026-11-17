# The four classes: Manifold, Field, Selection, Operator, typed as the Cortical Flow Studio types them (spec, 2026-11-17)

Status: documentation only, for decision. No code changes with this spec.

Decided by the maintainers (2026-11-17), and binding on everything below:
- The four classes, their typing, their coordinate systems and their measurements are the Cortical Flow Studio's, exactly: the classes, their types,
  the fields of each class, their relations and their resolution. Where an earlier nxr-neuro design of this round differs (its own five selection types,
  "a kernel is not a selection", a stored grid/irregular family, and others), the Studio's typing wins for the classes. Each difference is recorded in
  Appendix A with what happens to it.
- nxr-neuro PROVIDES signal processing, geometry processing and the MEG/EEG compute pipelines. `run` (a pipeline on a subject: find, stream, store) and
  `compute` (a node or a graph on data handed to it) stay as they are. What changes is their typing: node ports, a definition's inputs and outputs, and
  compute functions are typed by the four classes.
- The Studio will read nxr-neuro's YAML definitions to build them as TSL (GPU) graphs. A definition therefore carries the typing the Studio needs.

Sources. The Studio is cited by its repository-relative paths (`backend/src/entities/*.ts`, `backend/schema/model.sql`, `studio/src/...`, and its
`CLAUDE.md`, abbreviated **S-CLAUDE**); a bare `model.sql` is the Studio's `backend/schema/model.sql`. nxr-neuro is cited by its own paths. Line numbers are those of 2026-11-17.

---

## 1. Purpose and the rule

**Everything routes through four classes; there are no competing routes** (S-CLAUDE §1, lines 14–41). They are the ordinary primitives of any data
workflow, with identity and state added:

| class | IS | the plain word |
|---|---|---|
| **Manifold** | the SHAPE: it describes the axes | a shape |
| **Field** | the ARRAY: values defined on a Manifold | an array |
| **Selection** | the INDEXING: which part of a Manifold is in play | an index |
| **Operator** | the MAPPING: Field → Field | a function |

What they add over a bare array is **identity** (an `id`, a `name`: a thing can be found and tracked) and **state** (loaded, saved, calculating,
residency, the storage path) (S-CLAUDE lines 43–47). The same four ideas exist at every site the bytes visit: on disk (the store's arrays and their
attributes), in memory, and on the GPU; the verbs `load`, `write`, `claim`, `free` are copies between sites and nothing more (S-CLAUDE lines 61–85).
The classes route the copying: identity lets two consumers agree that they mean the same bytes at a different site.

**The membership test** for anything proposed on these classes (S-CLAUDE lines 104–117): *if removing the member would not change what bytes exist or
where they are, it does not belong on the class.* `shape`, element counts, a reference to storage and the values pass; a byte budget, a page concept or an
operator's context width fail. The Studio records one exception, a Field's display colormap, which is runtime-only, never stored, never in its row
(`backend/src/entities/field.ts:38–47`); nxr-neuro draws nothing and does not carry it.

Why nxr-neuro adopts them. Today nxr-neuro has no Field, Manifold or Operator object: there are io's row structs, the task API's outputs, the graph's
`PortType` with `cls = field`, and compute's bare matrices; three separate field-type enums; two registries no pipeline uses; and a Selection ported twice
(compute's and io's mirror). Each of those is a competing route. With one set of classes, a value made by `compute`, stored by `run`, read by the Studio and
drawn on its GPU is the same object at four sites, and the type a node's port declares is the type the Studio builds its TSL pass from.

---

## 2. The four classes, exactly as the Studio defines them

### 2.0 What every class shares: the entity

**An entity IS a row** (`backend/src/entities/entity.ts:1–16`, `:50–75`). It holds its `row` and exposes every column as a property of the same name;
relations are lookups BY ID through the model's identity map, never stored object references; `saved` says whether the row came from the database or was
made here and not created yet; `toJSON()` is the row.

- **Identity map** (`entity.ts:77–94`, `backend/src/entities/model.ts:1–10`): one collection per table, one object per key; adding a second object under
  the same key is refused. The key is the row's `id`, or the composite key of a detail row (`<selection>:<ordinal>`, `<topology>:<rank>`).
- **The loading contract `Loadable`** (`entity.ts:20–43`), implemented by the classes whose rows point at arrays (Field, Manifold, Operator, Selection, and
  a topology's Cells):
  - `load(selection?)`: read what the entity stores, the whole or the part a Selection is;
  - `loaded`, `loadedOn`: what is in memory now, and which Selection it covers (none: the whole);
  - `buffer`, `bufferOn`, `setBuffer`, `clearBuffer`: the device copy and what it covers;
  - `unload()`: drop both.
  A Space, a CoordinateSystem, a Dictionary and the containers (datastore, dataset, subject) are rows only and load nothing.
- **Status** (every class row's `status`: `pending` · `complete` · `deleting`; readers see only `complete`).
- **Provenance columns** on the class rows: `comment`, `source`, `producer_json`, `created_utc`, `created_by`, `modified_utc`, `modified_by`, and
  `acquisition_id` (the acquisition a node was made from or for).
- **Creation order** (`model.ts:1–10`, `:30–36`): a transient Manifold, Field, Operator or Selection is created bytes first, row last, its detail rows as
  children; measurements in bulk after the selections and fields they key on.

### 2.1 Manifold: the shape

**What it is.** A cell complex owned by a subject: its TOPOLOGY (how the vertices connect) + its GEOMETRY (the metric, in a coordinate system of a space) +
its SHAPE COUNTS, the f-vector (`backend/schema/model.sql:388–405`, table `manifold` at `:406–444`). A Manifold is a shape and nothing more: `points` a set,
`line` a 1-D vector, `image` a 2-D matrix, `volume` a 3-D matrix, `surface` an irregular mesh, `curve`, `graph`, `tetrahedral`, and `product` (S-CLAUDE
lines 26–29). Stored chunks are a Manifold too: time is a Line broken into smaller Lines.

**Attributes (the row).**

| column | meaning |
|---|---|
| `id`, `status`, `subject_id`, `name`, `path`, `label`, `acquisition_id`, `is_primary` | identity; `path` is null for a manifold no node stores (a product) |
| `type` | `points` · `line` · `image` · `volume` · `curve` · `surface` · `tetrahedral` · `graph` · `product` |
| `n_vertices`, `n_edges`, `n_faces`, `n_volumes` | the f-vector: primary data, known without reading any array |
| `topology_id`, `geometry_id` | the structure; a null geometry is a manifold with no metric and no place |
| `component_of_id`, `component_ordinal`, `from_index` | COMPONENTS: a hemisphere is a manifold of its own naming the union it belongs to, its order, and where its vertices start in the union |
| `derived_from_id`, `operator` | DERIVATION: an eigen axis names the manifold whose operator it diagonalises (and which operator); a frequency line the frequency manifold it samples |
| provenance | as 2.0 |

Its detail tables (`backend/src/entities/manifold.ts`):
- **Topology** (`model.sql:246–256`; `manifold.ts:19–45`): `kind` (`none` · `lattice` · `path` · `circle` · `simplicial` · `graph` · `product`), `rank`,
  `n_components`, `winding`, `closed`, `orientable`. Manifolds may share one (the cortex and its sphere).
- **Cells** (`topology_cell`, `model.sql:266–271`; `manifold.ts:64–92`): the STORED cells of one rank (edges `[e, 2]`, faces `[f, 3]`, tets `[t, 4]`), also
  that rank's boundary map read with signs, so d = ∂ᵀ needs no bytes. A lattice's cells are implicit. A mesh's edges, when not stored, are derived from its
  faces (`edgesOfFaces`, `manifold.ts:47–62`).
- **Geometry** (`model.sql:342–352`; `manifold.ts:94–109`): `form` `closed_form` (positions generated from dimension rows) · `stored` (a positions array
  at `positions_path`; the metric induced) · `product`; `coordinate_system_id` (null for a product: each factor carries its own); `direction_json`.
- **GeometryDimension** (`model.sql:367–379`; `manifold.ts:111–131`): one row per dimension in array order. A closed-form row is `n_vertices`, `spacing`,
  `origin`, `unit` (an inline Line: 1 / spacing on time is the sampling rate). A PRODUCT row instead names the factor manifold (`manifold_id`).

**Derived properties** (`manifold.ts:142–181`): `factors` (a product's, by ordinal), `components`, `componentOf`, `derivedFrom`, `fields`, `selections`;
`dimensions`, `indexShape` (a product's is its factors' concatenated), `rank`, `axis` (the one closed-form ordered axis, or none), `indexAt(value)` /
`valueAt(i)` (value ↔ index on a closed-form axis), `axisOffsetOf(factor)` (where a factor's first axis sits in an array on the product).

**Relations.** Fields lie on it (`field.manifold_id`); Selections index it (`selection.manifold_id`); Operators map from and to it; its geometry is written
in a CoordinateSystem of a Space. A subset of vertices that is not a component is a Selection, never a manifold (`model.sql:399–405`).

**State.** Its arrays (positions, faces, edges) are loaded on demand; loading on a Selection narrows the positions only (a sub-mesh with its own faces is
a component, not a slice), and a weighting selection is refused because it picks no vertices (`manifold.ts:230–278`).

### 2.2 Field: the array

**What it is.** Values on ONE manifold (a product for sensors × time), on one kind of its cells (`model.sql:681–701`, table `field` at `:702–736`;
`backend/src/entities/field.ts:1–6`). For most uses the Manifold can be read off the array's shape, and that is what indexes it.

**Attributes (the row).**

| column | meaning |
|---|---|
| `id`, `status`, `subject_id`, `name`, `path`, `label`, `acquisition_id`, `description` | identity |
| `kind` | what it is, open vocabulary: recording · image · eigenvalues · positions · orientations · index · … (a new kind is a row, not DDL) |
| `manifold_id`, `cell` | where the values sit: the manifold and the CELL KEY (2.2.1) |
| `value_type`, `n_components`, `unit` | what the values are: the STORAGE value type (3.1), components per cell, the unit |
| `data_type`, `layout` (`dense` · `csc`), `nnz`, `chunk_shape_json`, `codecs_json`, `fill_value_json` | how it is stored; `data_type = 'virtual'` is a VIRTUAL field (no array) |
| `of_field_id`, `function` | a SUMMARY of another field (`function`: `minmax`, or a group's `mean` over its contributions) |
| `of_operator_id` | a by-product of an operator (its eigenvalues) |
| `derived_from_id`, `by_operator_id` | a DERIVATION, and the operator that PRODUCED it from that input (D93) |
| provenance | as 2.0 (`producer_json` also carries the Studio's algebra, 3.1) |

Detail: **FieldContribution** (`model.sql:749–757`; `field.ts:16–27`): what a group field accumulates (subject, source field, weight, ordinal).

**Derived properties** (`field.ts:49–166`): `manifold`, `domain` (a product's factors, else the one), `ofField`, `ofOperator`, `derivedFrom`, `byOperator`,
`summaries`, `selections` (the Selections ON it: its chunk grid, flags, events), `contributions`; `shape` (its cells' counts per factor, then the components),
`elements`, `orderedAxis`; **`chunks`**, the chunk grid found BY DEFINITION, never by name: the `spans` Selection on the ordered axis whose members tile it
contiguously in the stored chunk width (`field.ts:99–109`, `:130–166`); **`envelope`**, the closed-form tiling with array-backed `min`/`max` measurements
(`field.ts:110–128`); `virtual` (`field.ts:210–215`).

**State.** `values` (CPU), `loadedOn`, `loadedSlice`, `buffer`, `bufferOn`; `setValues(v, slice?)` for a transient field's bytes; `sliceOf(selection)`
(a Selection on the field's manifold or on one factor of its product, which must resolve to a crop); `load(selection?)`; `loadChunk(k)`
(`field.ts:168–234`). A virtual field refuses `load`: its values are computed from `derived_from` through `by_operator`.

#### 2.2.1 Cells

A manifold is a chain complex; where a Field sits is its CELL KEY (`backend/src/db/cells.ts:1–15`; S-CLAUDE lines 375–385): `'0'` vertices · `'1'` edges ·
`'2'` faces · `'3'` volumes; `'1*'` a dual cell (the Hodge star's image, primal k ↔ dual n − k); on a product one rank per factor, `'0,0'` (source currents
on cortex × time), `'0,1'` (their time derivative). The default `'0'` is the vertices on a product too. The count of cells of a key is the f-vector entry
(a product's: the product of its factors'). Writers refuse a key that does not fit its manifold (`cells.ts:59–64`, `checkCell` `:100–104`).
**A Selection is a CHAIN and a Field a COCHAIN** (D94): a measurement is their pairing.

### 2.3 Selection: the indexing

**What it is.** A named part of a manifold, with a code, measurements and its detail rows; or a POSITION on a space (`model.sql:486–531`, table
`selection` at `:532–577`; `backend/src/entities/selection.ts:1–9`). It is stateful, identified and carries its own metadata (S-CLAUDE lines 34–36).

**Its types** (the schema's CHECK, `model.sql:538`; their descriptions `:489–519`):

| type | described by | resolves to (`selection.ts:177–227`) |
|---|---|---|
| `point` | `center` (an element); on a SPACE, `position` in `coordinate_system_id` | a crop of one index |
| `span` | RECTANGULAR: its corners, `[start, stop)` on each constrained axis (`selection_extent` rows); on a product one extent per factor (a box); on an axis with no closed form (points) the extent is element indices | a crop |
| `distance` | `center`, `radius` in `unit` under the manifold's metric (an interval, a disc, a ball; on a surface the level set of the heat distance) | not resolved by the Studio's entity (the toolbox evaluates it) |
| `spans` | a TESSELLATION of a line: members as `selection_span` rows (events), or in CLOSED FORM `params_json.tiling = {origin, width, hop, count}` (a constant-Q level, the chunk grid); a member's ordinal is its code | member by member: `at(k)` |
| `indices` | the picked elements (its array) | indices (a gather) |
| `kernel` | a rule: `weight_fn` + `params_json` at the values in use | weights |
| `set` | a GROUP of Selections (member rows: `member_of_id`, `member_ordinal`), or a PARTITION stored as one array, a code per element (`array_path` + `dictionary_id`); its members are "the elements whose code is k" (D48) | indices (its array) |
| `ladder` | top level, with a dictionary; its members are LEVELS, whose codes are read with the ladder's dictionary | through a level |

Every selection on a manifold is a set of CELLS of one kind (`cell`, default `'0'`); a `span` or `spans` is never on vertices (`cell <> '0'`: a span on a
Line is its edges) (`model.sql:573–576`). A Selection is on a manifold OR on a space, never both (D95; `:568–572`).

**Attributes (the row)** (`model.sql:532–562`): `id`, `status`, `subject_id`, `name`, `path`, `type`, `label`, `acquisition_id`, `description`;
`manifold_id` or `space_id`; `cell`; `of_field_id` (the Field it belongs to: a recording's flags, its chunk grid); `member_of_id`, `member_ordinal`;
`code` (what this selection IS, as an integer, read with a dictionary); `dictionary_id`; `n_elements`, `n_members`; `array_path`; `center`, `position`,
`coordinate_system_id`, `radius`, `unit`; `weight_fn`, `params_json`; created/modified.

Its detail rows: **SelectionExtent** (a span's corners per axis, `model.sql:602–608`; `selection.ts:26–32`), **SelectionSpan** (a spans set's members,
`[start, stop)` and `code`, `model.sql:671–678`; `selection.ts:34–41`), the ELEMENT ARRAY (a partition's codes or a subset's picks, at `array_path`;
`selection_element` rows are no longer written, `model.sql:583–593`), and its **measurements** (2.7).

**Derived properties** (`selection.ts:69–175`): `manifold`, `space`, `coordinateSystem`, `vertexOn(line)` (where a position falls on a Line sampling its
space), `ofField`, `memberOf`, `members`, `dictionary`, `entry` (what its `code` means: its SET's dictionary first, its own only when it is not a member),
`extents`, `spans`, `measurements`, `measurement(measure, ofField)`, `params`, `tiling`, `at(k)` (member k as a `span` with the derived id `<id>#<k>`, by
arithmetic for a closed form, by lookup for rows), `count`, `indexAt(value)`.

**Resolution** (`selection.ts:20–24`, `:177–227`). Three arms, three efficient realisations of ONE idea, weighting a Field (S-CLAUDE lines 38–41):
- `crop`: index ranges per axis (a hard window applied by not reading the rest);
- `indices`: the picked elements (a delta comb applied by gathering);
- `weights`: `weight_fn` + `params` (the general form; the toolbox evaluates it).

**State.** A closed-form selection (span, point, spans, kernel) stores no array: its row and detail rows ARE it, so it is always loaded; an array-backed
one is loaded once read (`selection.ts:246–265`). A Selection is never narrowed by another.

**Products of partitions** (D100; `backend/src/entities/product.ts:1–17`). One Selection whose members are every combination of its factors' members: a
`set` on the product of the factors' manifolds, with no array, its members a RULE in `params_json.product = [{selection, members}]`, its cell key the
factors' joined (`'0,1'`), `n_members` their product. The code is mixed radix, FIRST factor fastest: code = c₀ + n₀·(c₁ + n₁·(c₂ + …))
(`productCode`, `productCodes`, `product.ts:32–51`). A factor's member count (`factorMembers`, `:53–73`): a span or point has one; a spans set its count;
a set its codes' count (nested products multiply); any other type is not a partition and is refused. A factor's member per element (`factorCodes`,
`:75–98`), −1 in none. The JOINT partition of space × time is a product whose time factor's radix is rounded up to a power of two, so
`code = (tile << B) | frame` (`backend/src/db/atlas.ts:11–13`).

### 2.4 Operator: the mapping

**What it is.** Field → Field: a stored matrix from one manifold to another, or a transition map between two coordinate systems, or an ANALYTIC operator
with no bytes (`model.sql:760–782`, table `operator` at `:783–831`; `backend/src/entities/operator.ts:1`).

**Attributes (the row).**

| column | meaning |
|---|---|
| `id`, `status`, `subject_id`, `name`, `path`, `kind`, `label`, `acquisition_id`, `description` | identity; `kind` open vocabulary (inverse kernel · forward · eigenvectors · mass · projection · transition · covariance · whitener · …) |
| `from_manifold_id`, `to_manifold_id` | the two ends (the codomain may be null where its row space has no manifold row yet) |
| `from_cell`, `to_cell` | the cell keys of the two ends (a differential operator changes the cell: d₀ is `'0'` → `'1'`) |
| `from_selection_id`, `to_selection_id` | a restriction of the domain or codomain (the channels a kernel maps) |
| `from_value_type`, `to_value_type` | the storage value types it reads and writes |
| `from_coordinate_system_id`, `to_coordinate_system_id` | a TRANSITION MAP between two systems (both or neither) |
| `layout` | `dense` · `csc` · `csr` · `analytic` (no bytes: evaluated by its node from the geometry or a closed form) |
| `method`, `params_json` | how an analytic operator is evaluated (`fem-tangential` · `fem-ambient` · `dec` · `fft`) and its parameters |
| `n_rows`, `n_cols`, `nnz`, `data_type`, `chunk_shape_json`, `codecs_json`, `fill_value_json` | the shape (cells × components of the two ends) and storage |
| `derived_from_id` | a projection derived from another operator |
| `inverts_id`, `inverse_kind` | THE PAIR (D106): the operator this one inverts, and how: `exact` · `left` · `regularized` · `adjoint` |
| provenance | as 2.0 |

Detail: **OperatorFactor** (`model.sql:847–856`; `operator.ts:12–22`): a COMPOSED operator's factors in application order, each an operator (possibly
transposed) or a GAIN, a `kernel` Selection on a spectral manifold.

**Derived properties** (`operator.ts:32–80`): `fromManifold`, `toManifold`, `fromSelection`, `toSelection`, `fromCoordinateSystem`,
`toCoordinateSystem`, `analytic`, `params`, `produced` (the Fields it made, `by_operator_id`), `summaries` (its by-products: eigenvalues), `inverts`,
`inverseKind`, `inverseMethod` (an inverse kernel's method, `params_json.inverse`), `inverses`, `factors`, `shape`, `chunkShape`. Its **kind** is read off
its rows, never off its name (`backend/src/entities/operatorKind.ts:1–22`, `:78–99`):

| kind | rule |
|---|---|
| `placement` | a transition between coordinate systems |
| `differential` | from and to are ONE manifold (the same row, or components of one union), within or between cell ranks |
| `spectral` | a manifold and a spectral line DERIVED from it (an eigenbasis's synthesis or analysis, the DFT, Welch) |
| `linking` | two different manifolds, neither derived from the other (the inverse kernel, the forward model), or a composition one of whose factors is linking |

**State.** `values` (dense, f32), `loadFloat64()` (the stored precision, for an eigenbasis or an affine), `loadColumns(k)` (the first k modes),
`loadSparse()` / `setSparse()`, `sliceOf(selection)` (a crop on the TO manifold picks rows, on the FROM manifold columns, only where there is one row or
column per vertex), `buffer`; an analytic operator refuses `load` (`operator.ts:82–197`).

**Pairs and chains are checked** (`backend/src/db/pairs.ts:1–40`; S-CLAUDE lines 430–436): an inverse's ends are its forward's swapped (manifolds,
cells, value types); a composition's factors meet end to end (a gain acts in place); a field's lineage `derived_from` + `by_operator` must meet; W after
G is flagged unless declared a round trip; a linear map after a magnitude is flagged.

### 2.5 Space and CoordinateSystem: the charts

**Space** (`model.sql:273–297`; `backend/src/entities/space.ts:1–14`): an ANALYTIC manifold, a parametric geometry with no discretisation: `name`
(which copy: head, sphere, time, frequency, a spectral line), `geometry` (`line` · `circle` · `plane` · `sphere` · `torus` · `volume`), `rank`,
`parameters_json` (a radius), `description`. A lattice SAMPLES it, a mesh is PLACED in it, an analytic field is written on it. A row anchors
comparability: two systems of one space are related by a transition, two spaces are not. Derived: `coordinateSystems`, `reference`, `parameters`.

**CoordinateSystem** (`model.sql:299–325`; `space.ts:16–24`): a chart in the mathematical sense, a coordinate map on a space with a unit: `space_id`,
`name` (`scs` · `ncs` · `voxel:<volume>` · `clock` · `linear` · `log10` · `eigenvalue` · `wavelength` · `scene`), `unit`, `is_reference` (one per space),
`transition` (`identity` · `affine` · `log` · `power` · `operator`) and `transition_json` (its parameters FROM the reference). A non-isometric system
(log, power) presents the space in other numbers and changes no metric. Time → frequency is not a chart change; it is an operator on Fields. Derived:
`space`, `transitionParams`. Every manifold's positions are numbers in one system; a Selection on a space is a position in one.

### 2.6 Dictionary: the labels

**Dictionary** and **DictionaryEntry** (`model.sql:450–484`; `backend/src/entities/dictionary.ts:1–28`): one table of entries translating a Selection's
codes (an atlas, a channel family, a ladder's levels, quality flags, a user's labels). The header has `id`, `name`, `scope` (`app` · `dataset` · `subject`
· `user`), `dataset_id` / `subject_id`, `description`. An entry is keyed `(dictionary_id, code)` and gives `name`, `color_json`, `measurement` +
`measurement_unit` (a threshold, a bucket width, a band edge), `function` + `params_json` (an aggregation, a kernel), `attributes_json` (a constant-Q
band's edges, window, hop and tile counts), `ordinal`. Derived: `entries`, `entry(code)`, `selections`.

### 2.7 Measurement: what was measured on a selection

**SelectionMeasurement** (`model.sql:610–624`, table `:625–651`; `selection.ts:43–52`): one value per (selection, member code, field, measure, within):

| column | meaning |
|---|---|
| `selection_id`, `code` | the selection, and the MEMBER the value is for (null: the selection as a whole) |
| `of_field_id` | the field measured (null: a geometric measurement of the selection itself: area, vertices) |
| `measure` | a name registered in the app-wide `measure` dictionary (`mean` · `max` · `mean_square` · `area` · …) |
| `value`, `weight` | the value; what it counts for in a roll-up (null = 1) |
| `within_selection_id`, `within_code` | the value over member `code` RESTRICTED to member `within_code` of another selection (a channel group, a parcel) |
| `unit`, `computed_utc`, `computed_by` | |
| `array_path`, `component` | ARRAY-BACKED (D96): one row per measure, the values an array `[rows × members × components]` in the store |
| `of_statistic_id`, `of_index` | the value measures slice `of_index` of a statistic (a PSD's peak) |

Values are stored at the FINEST level of a tiling; a coarser view is the dyadic roll-up `code >> shift` under the measure's reduction (`model.sql:617–619`).
A ratio is derived at read (`mean = sum / weight`, D132; `backend/src/db/atlas.ts:9–10`).

### 2.8 Statistic: what is computed from a partition's members

**Statistic** (`model.sql:1156–1193`; `backend/src/entities/statistic.ts:1–44`): a summary of a partition's members over a span of a parent (a Welch PSD,
a cross-spectrum, a histogram): `measure`, `method`, `value_type`, `n_components`, `of_field_id` or `of_statistic_id`, `partition_selection_id` (the
partition it is computed from: provenance), `span_json`, `n_members`, `members_skipped`, `dof`, `axes_json` (`[{name, manifold_id | selection_id | edges |
values_path, size, unit}]`), `components_json`, `unit`, `scaling_json`, storage columns, `options_json`, `producer_json`. Derived: `ofField`,
`ofStatistic`, `partition`, `measureSpec` / `methodSpec` (the registry), `axes`, `layout`, `axisSelection(k)`; `load()` reads its values checked against
its axes and value type. (The table originated in nxr-neuro and is now the Studio's.)

### 2.9 The model: the identity map and the bundle

One subject's rows come as a **Bundle**, every table's rows exactly as stored, a select per table, no join flattened (`backend/src/entities/rows.ts:24–58`);
the classes are built from it and nothing else. Arrays are read by an **ArrayReader** (a path in the subject's store, an optional slice per axis, an
optional cast; `rows.ts:60–61`). A model writes through a **ModelWriter**: create an array, put a region, post a row with its children, post
measurements in bulk (`model.ts:30–36`).

### 2.10 The C++ form in nxr-neuro

One class per Studio class, with the Studio's names, columns and derived properties. Each class holds its row (the columns, generated from
`io/schema/model.sql` as `io/include/nxr/io/records/rows.h` is today by `tools/gen_record_rows.py`) and the in-memory state of 2.0. The architecture
decides where each part lives:

| part | where | why |
|---|---|---|
| the classes: `Manifold` (+ `Topology`, `Cells`, `Geometry`, `GeometryDimension`), `Field` (+ `FieldContribution`), `Selection` (+ `SelectionExtent`, `SelectionSpan`), `Operator` (+ `OperatorFactor`), `Space`, `CoordinateSystem`, `Dictionary`, `DictionaryEntry`, `Measurement`, `Statistic`, and `Model` (the identity map) | `nxr::model`, a header-mostly target with no file access and no maths beyond index arithmetic (open question B1 decides its folder) | they are data with identity; compute, io, the graph and the bindings all name them |
| the index arithmetic the Studio puts on the classes: `indexShape`, `indexAt` / `valueAt`, `axisOffsetOf`, `shape`, `sliceOf`, `resolve` (crop and indices arms), `at(k)`, `count`, `indexAt`, `chunks` (by definition), `productCode` / `productCodes`, `factorMembers`, `factorCodes`, `operatorKind`, `pairProblems` / `chainProblems`, `algebraOf`, cell keys | `nxr::model` | pure functions of rows and small arrays; the Studio has them on the entities |
| the maths that makes or uses a class: forms, transforms, filters, geodesic distance (`resolve_distance`), the makers of selections (tilings, ladders, dyadic tiles, products, detections), operator assembly | `compute/`, `meeg/` | functions of matrices that take and return the classes (4.3) |
| loading and storing: `read` a subject's Bundle into a `Model`, `load(entity, selection?)` (fills `values` / `positions` / an element array, sets `loadedOn`), `create` (bytes first, row last), measurements in bulk, the chunk service | `io/` | io never computes; it fills and writes the classes' state |
| `buffer` / `bufferOn` | `nxr::model`, an opaque handle | nxr-neuro has no device today; the slot is where a host's GPU copy is recorded (the wasm binding that serves the Studio) |

Values keep the stored precision (float32 or float64), as the Studio's `Operator.loadFloat64` does where precision matters (`operator.ts:165–177`);
`values` is therefore a float32 or a float64 array, never a blanket double.

---

## 3. Typing

### 3.1 Value types: the algebra and the storage

Two vocabularies, ONE mapping (`backend/src/valueType.ts:1–28`):

- the **ALGEBRA** the Studio speaks: ℝ `real` (a number) · ℂ `complex` (a tangent vector in the surface) · ℍ `quaternion` (a number w and a vector x
  out of the surface) (`studio/src/engine/toolbox/differential/fieldForms.ts:4–6`, `:26`; `studio/src/studio/ports.ts:15–16`);
- the **STORAGE** value type of the schema (`field.value_type`, `operator.from_value_type` / `to_value_type`, `statistic.value_type`;
  `model.sql:714`): `scalar` · `integer` · `complex` · `vector2` · `vector3` · `matrix3` · `quaternion` · `hermitian3`.

| algebra | written as storage (`STORAGE`, `valueType.ts:10–14`) | values a vertex in memory (`widthOf`, `fieldForms.ts:76–77`) |
|---|---|---|
| ℝ | `scalar` | 1 |
| ℂ | `vector3` (EMBEDDED tangent vector), with the algebra in `producer_json.studio.type` | 3 |
| ℍ | `quaternion` | 4 (w x y z) |

Reading the algebra of a stored value (`algebraOf`, `valueType.ts:19–28`): the row's `producer_json.studio.type` when it fits the storage; else a
`vector3` registered as a tangent field is ℂ; else the old rule (`valueType.ts:17`): `scalar`, `integer`, `matrix3` → ℝ; `complex`, `vector2` → ℂ;
`vector3`, `quaternion` → ℍ. So a free-orientation current stored as `vector3` reads as ℍ with w = 0, as the Studio's kernel output is
(`studio/src/studio/engine/ingest.ts:221`). `hermitian3` (a complex 3 × 3 Hermitian matrix a cell, `[.., 3, 3, 2]`) is a storage type of statistics and
fields with no algebra of its own (`backend/src/db/statistic-values.ts:34`).

nxr-neuro adopts both vocabularies and the one mapping: one `Algebra` enum, one `StorageType` enum, `STORAGE`, `algebraOf`, `widthOf`. Values outside
them (a complex value in TIME, the analytic signal; a complex vector3; the monogenic signal) are open question B3.

### 3.2 Forms

A form is a quantity drawn from a field, of its own algebra (`fieldForms.ts:47–67`, `FORMS`):

| algebra | forms (id → out) |
|---|---|
| ℝ | `f` → ℝ |
| ℂ | `abs`, `div`, `curl`, `div_tan`, `div_3d`, `div_curv`, `div_2H`, `curl_tan`, `curl_3d`, `curl_curv` → ℝ; `v`, `Jv` → ℂ |
| ℍ | `speed`, `absq`, `nx`, `w`, the div/curl forms of xₜ → ℝ; `xt` → ℂ; `x` → ℍ |

A form draws in the SCALAR slot when its output is ℝ, in the VECTOR slot otherwise (`fieldForms.ts:69–74`). nxr-neuro's `compute/include/nxr/forms.h`
is already the line port of this file; it keeps the Studio's ids and names and takes its algebra from 3.1.

### 3.3 Cells

The cell key of 2.2.1 types where a Field's values sit, which cells a Selection indexes, and the two ends of an Operator. Its parse, format, primal rank
and count are `backend/src/db/cells.ts` (`parseCell`, `cellParts`, `formatCell`, `primalRank`, `cellCount`, `cellMisfit`), ported verbatim into
`nxr::model`.

### 3.4 Products of manifolds

A product is a manifold of `type = 'product'`, topology `kind = 'product'`, geometry `form = 'product'` with one `geometry_dimension` row per factor
naming the factor manifold (D37; `model.sql:356–366`). It has no path and no stored cells; its `indexShape` is its factors' concatenated, its cells are
products of its factors' cells (a cell key has one part per factor), and a Field on it is a Field on all its factors at once. The Studio's input port is
the same (`ports.ts:89–95`: `factors` in order). A Selection on a factor of a product slices the Field at the factor's offset (`field.ts:195–208`).

### 3.5 The operator's ends

An Operator is typed by its two ends: (from manifold, from cell, from value type) → (to manifold, to cell, to value type), plus its `layout`, `method`
(analytic) and a restriction (`from_selection_id`, `to_selection_id`). Its kind (placement · differential · spectral · linking) is DERIVED from those
(2.4), never declared. A pair and a chain are checked by `pairProblems` and `chainProblems` (`pairs.ts`).

### 3.6 The operator ↔ field-type pairing, as the Studio gates it

The Studio has no operator type table; the pairing is gated where an operator is applied:

| gate | rule | source |
|---|---|---|
| applying an operator to a field | the operator maps FROM the field's manifold; it takes one column an element (`colsPerElement` 1); the field is ℝ; the output is ℍ when the operator has 3 rows an element (a free vector, w = 0), ℝ when it has 1 | `studio/src/studio/engine/ingest.ts:212–224`; `ports.ts:169–193` |
| a stored kernel's rows an element | `to_value_type = vector3` → 3, else 1 | `studio/src/adapters/nxr/index.ts:371` |
| a kernel's columns | its `from_selection` names the channels; columns that do not index the members are refused | `ports.ts:328–337`; `studio/src/adapters/nxr/index.ts:355–369` |
| a spectral filter on the surface | ℝ by the Laplace–Beltrami operator, ℂ by the connection Laplacian, ℍ by the relative Dirac | `studio/src/engine/compute/runtime/surfaceKit.ts:1368–1376` (capabilities `:459`) |
| an anisotropic kit | ℝ only; ℂ and ℍ refused | `surfaceKit.ts:546`, `:572` |
| a linear map after a non-linear step | flagged (D99, D106) | S-CLAUDE lines 424–436 |

nxr-neuro's graph checker applies exactly these gates, from the classes' types, before any data flows (4.4).

### 3.7 Selection types, resolution and members

The types are the schema's eight (2.3: point, span, distance, spans, indices, kernel, set, ladder); the resolution is the three arms (crop, indices,
weights); a member is reached by `at(k)` (spans), by `members` (a set of rows), by its code (a partition: the elements whose code is k), or by
`productCodes` (a product). Whether a selection is a PARTITION is not stored: it is read off its type and rows (a partition `set`; a `spans` tiling whose
members tile its line, found by definition as `Field.chunks` does). Appendix A lists what this changes in nxr-neuro.

### 3.8 Measurements and statistics

A measurement is typed by (selection, member code, field, measure, within) and the measure's registry entry (its unit and reduction). A statistic by
(measure, method, value type, axes, of). A port of either names these, never an array shape alone.

---

## 4. How `compute` and `run` adopt it

`run` still finds, streams the stated time range chunk by chunk, and stores; `compute` still applies a node or a graph to data handed to it. Both now
speak the classes.

### 4.1 Node ports typed by the classes

A node's port declares its CLASS and that class's type, nothing else:

| class | a port's type |
|---|---|
| Manifold | `type` (points · line · surface · volume · curve · graph · product), for a line its role (`time` · `spectral` · `parameter`, `ports.ts:72`), for a product its factors |
| Field | its manifold (a term: a variable, an input's manifold, or one the node makes), `cell`, algebra, storage `value_type`, `n_components`, `unit`; streamed or not (whether its time factor is walked block by block) |
| Selection | `type` (one of eight), its manifold (or its space and coordinate system), `cell`, and the resolution arm the node takes (crop · indices · weights · members) |
| Operator | from (manifold, cell, value type) → to (manifold, cell, value type), `layout`, `method` for an analytic one |
| Measurement | the selection it is over, the field it is of, the measure (a registry name), `within` |
| Statistic | measure, method, value type, axes, what it is of |

Parameters stay parameters (quantities with units and charts; `compute/include/nxr/quantity.h`). The node signature machinery (manifold variables,
unification, the moves; `pipelines/graph/include/nxr/graph/signature.h`) is kept: variables range over Manifold ids; the moves keep their meaning.

### 4.2 Definitions: inputs and outputs in the classes' terms (what the Studio needs)

To build a definition as a TSL graph, the Studio needs, for every port, what its own ports carry (`ports.ts`) and what its device chain needs
(S-CLAUDE lines 637–666): which buffer is a Field on which manifold (and which factor is the time Line it streams), its algebra and width, which matrix
is an Operator between which manifolds and how many rows an element, which Selection is a uniform or a small index buffer, and which steps are linear.
A definition therefore declares, beside the find clauses `run` uses (`where`, `each`, `hint`, `required`, unchanged):

```yaml
inputs:
  recording:
    class: field
    on: {product: [{type: points, as: sensors}, {type: line, role: time}]}
    cell: "0,0"
    algebra: real
    value_type: scalar
    where: {name: $recording}
  kernel:
    class: operator
    from: {manifold: recording.on.0, cell: "0", value_type: scalar}
    to:   {manifold: {type: surface, as: cortex}, cell: "0", value_type: vector3}
    layout: dense
    restricts: from                       # the kernel's own channels: from_selection_id (the Studio's `columns`)
    where: {name: $kernel}
  bands:
    class: selection
    type: spans
    on: {space: frequency, coordinate_system: hz}
    resolution: crop
    where: {name: $bands}

steps:
  chunks:
    over: recording
    do:
      - {node: apply_kernel, in: {x: z, kernel: kernel}, out: sources}

outputs:
  sources:
    class: field
    on: {product: [kernel.to, recording.on.1]}
    cell: "0,0"
    algebra: quaternion                   # 3 rows an element: ℍ with w = 0 (the Studio's application gate)
    value_type: vector3
    derived_from: recording
    by_operator: kernel
```

Each registered node additionally states, for the Studio: its place in the device chain where it has one (`members` · `filter` · `transform` · `forms` ·
`measure`, the order of `studio/src/engine/toolbox/signal/chain.ts`), whether it is LINEAR (a linear map may not follow a non-linear node, D106), its
halo, and its output shape on a streamed input (the Studio's `xt`, `xft`, `xf`). Most of a definition's types are INFERRED from its inputs and the nodes'
signatures; `--dry-run` prints them, and a generated JSON schema of definitions is published with the release for the Studio to read. Where the Studio
has no pass for a node, the node runs on the CPU (nxr-neuro) and its output enters the GPU chain as a Field.

### 4.3 Compute functions take and return the classes

A node function takes Fields, Selections, Operators and Manifolds and returns them; the maths inside stays a function of matrices.

- **A Field can always be built from a matrix made anywhere.** `Field::from_array(values, shape, algebra)` makes a transient Field (`saved = false`) on a
  transient Manifold read off the shape (a Line of n samples, Points of n members, their product), as the Studio's chart API inlines a Manifold as a bare
  shape (S-CLAUDE lines 49–59). So the architecture's test, "can each compute function be called with a matrix from anywhere?", still holds: the class
  wraps the matrix, and nothing in it says where the matrix came from.
- A compute function reads the class's TYPE (algebra, cell, manifold shape, selection type) and its VALUES; it never reads `path`, never loads, never knows
  `saved`. What it makes is a transient class with `derived_from` / `by_operator` set, which io may later create.
- Reductions remain measurements over Selections: a function that reduces returns Measurements (or a Statistic), never a Field of fewer elements. The
  pointwise rule and the cardinal rule of `ARCHITECTURE.md` are unchanged.
- The makers of selections (tilings, ladders, dyadic tiles, products, detections) are compute functions whose OUTPUT is a Selection of one of the eight
  types; they are not part of the class.

### 4.4 The type checker re-based on the classes

`pipelines/graph/include/nxr/graph/types.h` keeps its role (what a port carries, checked before any data) and its machinery (terms, unification,
`print` / `parse_type`), with the classes' types as its content:

- `PortType.cls` becomes the class (manifold · field · selection · operator · measurement · statistic; `constant` stays for parameters; `partition`
  goes: a partition is a Selection whose type and rows make it one);
- `ValueType` becomes the algebra + storage pair of 3.1;
- `Domain` becomes a Manifold term (a product's factors) + a cell key;
- `SelectionShape` becomes the selection type of 3.7;
- an Operator port carries both ends (3.5) and its derived kind;
- the checker runs the Studio's gates (3.6) and its pair and chain rules (`pairs.ts`), and the D106 lineage rule from each port's lineage.

---

## 5. What it replaces in nxr-neuro

| today | replaced by |
|---|---|
| `compute/include/nxr/field_registry.h` (`FieldDescriptor`: domain, bundle, field type, n-form, representation, gauge; the conversion graph) | the Field class: `cell` (the n-form and the domain), the algebra and storage value type (bundle and representation), the surface's gauge and a ℂ field's frame (`ports.ts:33–43`, `:149`); a conversion is an Operator (an analytic one, `layout: analytic`, `method`) |
| `compute/include/nxr/operator_registry.h` (`OperatorVariant`: bundle, holonomy, order, role, field type, domain, gauge, coupling, …) | the Operator class: its ends (manifold, cell, value type), `layout`, `method`, `params_json`, its pair and its factors; its kind derived (2.4); the pairing gates of 3.6. The catalogue's descriptions move to `docs/reference.md`; the Studio removed its own operator registry the same way |
| the three field-type enums: `nxr::forms::FieldType` (`forms.h:33`), `nxr::manifold::registry::FieldType` (`operator_registry.h:16`), `nxr::graph::ValueType` (`types.h:30`) | ONE `Algebra` (ℝ ℂ ℍ) + ONE `StorageType` (the schema's eight) + `algebraOf` (3.1) |
| compute's Selection (`compute/include/nxr/selection/selection.h`: five types, `Family`, `Grid`, `member_of`, `member_codes`) | the Selection class (2.3), eight types, the schema's columns; the makers stay in compute as functions producing it (Appendix A) |
| the resolved forms (`compute/include/nxr/selection/resolved.h`: `Rule`, `Intervals`, `Elements`, `MemberOfElement`, `ProductForm`) | the Studio's resolution and member access: a closed-form `tiling` with `at(k)` / `indexAt` (Rule); `at(k)` resolved to a crop, per member (Intervals); `indices` resolution per member (Elements); `factorCodes` (MemberOfElement); `productCode` / `productCodes` over the factors' codes (ProductForm). The array kernels compute loops over keep their implementations under the Studio's names |
| io's Description (`io/include/nxr/io/selection_description.h`) and the pipelines' `selection_bridge.h` | gone: io loads and stores the Selection class itself |
| io's selection entity (`io/include/nxr/io/selection.h`) and store (`selection_store.h`) | io's `read` / `load` / `create` for the classes (2.10); the store's row functions stay inside io |
| the graph's parallel class labels: `PortClass` (with `partition`), `ManifoldKind` (with `members`), `Role`, `Manifold::Form` (`restricted`, `members`, `derived`), `SelectionShape`, `CoordinateSystem` (`types.h:31–154`) | the classes' types (4.4): `ManifoldKind` → `manifold.type`; `Role` → a line's role; `restricted` → a Selection beside a manifold (an operator's `from_selection_id`; a Field is always on ONE manifold); `members(P)` → a Measurement over P (there is no regions manifold, `operatorKind.ts:40–41`); `made` → a transient Manifold (`saved = false`), a spectral line with `derived_from_id`; `derived` (a glued surface) → a manifold of its own related by an operator; `CoordinateSystem` → the class (2.5) |
| `compute/include/nxr/quantity.h` `Chart` | the CoordinateSystem class (its `transition`, `transition_json`, `unit`); the quantity parser stays |
| compute's geometry kit `nxr::manifold::Manifold` (`compute/include/nxr/compute.h`: a mesh and its cached operators) | renamed `SurfaceKit`, the Studio's name for the same thing (`studio/src/engine/compute/runtime/surfaceKit.ts`), so `Manifold` names the class |
| io's task outputs (`FieldOut` and the like) and the pipelines' `Value` with a class label | the classes |

---

## 6. Migration plan (no code now)

Each step is one batch on one branch, a PR when the batch is done; synthetic tests only; the full suite at the end of the round.

1. **Decisions.** The open questions of Appendix B are answered; the dependency table in `ARCHITECTURE.md` is amended for the classes' target (B1).
2. **The classes' target.** Generate the row columns from `io/schema/model.sql` into `nxr::model` (extend `tools/gen_record_rows.py`; io's repositories keep
   `from_row` and the SQL). Add the classes, the identity map, `Loadable` state, and the index arithmetic of 2.10, with tests ported from the Studio's
   own (`backend/src/entities/product.test.ts`, `operatorKind.test.ts`, `statistic.test.ts`, `derived.test.ts`, `model.test.ts`) as synthetic goldens.
3. **Value types.** One `Algebra`, one `StorageType`, `STORAGE`, `algebraOf`, `widthOf`; `forms.h` and the graph take them; the three enums are deleted.
4. **Selection.** The compute Selection becomes the class (eight types); the makers return it; `resolved.h` is re-expressed under the Studio's names;
   io's Description and the bridge are deleted. The stored changes of Appendix A (partition arrays of codes, levels as member rows, no `selection_factor`,
   no `params_json.family`, no elements grid) get a written migration of stored selections, run on development data only.
5. **Field, Manifold, Operator, Space, CoordinateSystem, Dictionary, Measurement, Statistic.** io's task API returns and takes them: `find` gives a class
   with its row, `open` loads into it, `commit` creates in the Studio's order. `compute.h`'s kit is renamed `SurfaceKit`.
6. **The checker.** `types.h` re-based (4.4); node signatures rewritten in the classes' types; the gates of 3.6 and `pairs.ts`'s rules as checker rules.
7. **Definitions.** Typed ports in every definition in `pipelines/definitions/` (most inferred; 4.2); each node's chain place, linearity, halo and output
   shape registered; the JSON schema of definitions generated and published; `--dry-run` prints the classes' types.
8. **Registries retired.** `field_registry.h` and `operator_registry.h` deleted once their content is in the classes and `docs/reference.md`.
9. **Faces.** The Python and MATLAB bindings expose each class as its row (a dict / struct with the column names) plus its arrays; `compute` accepts a bare
   array and builds the class (4.3).
10. **Docs.** `ARCHITECTURE.md` ("Decided structure") and `CONTRIBUTING.md` state the four classes as the typing of every port; the earlier specs this
    supersedes (`2026-11-17-type-system.md` §14, `2026-11-17-selection-module.md` §0 items 1, 3, 12) carry a pointer to this spec.

---

## Appendix A. This round's nxr-neuro Selection design and the Studio's, difference by difference

Sources: nxr-neuro `docs/specs/2026-11-17-selection-module.md` (§0, §8, §9), `docs/specs/2026-11-17-consolidation.md` (§1, §6),
`compute/include/nxr/selection/selection.h`, `io/include/nxr/io/selection_description.h`, `pipelines/graph/include/nxr/graph/types.h`; the Studio as in 2.3.

| # | topic | this round's nxr-neuro design | the Studio | under "exactly as the Studio" |
|---|---|---|---|---|
| A1 | the types | five: `point`, `span`, `distance`, `indices`, `set`; `spans` and `ladder` only stored row types, mapped to a set (`selection-module.md` §0 item 1; `selection.h:56`) | eight: `point`, `span`, `distance`, `spans`, `indices`, `kernel`, `set`, `ladder` (`model.sql:538`) | eight. `spans` and `ladder` are types of the class again |
| A2 | kernel | "a kernel is not a selection: it is the weights a measurement uses" (`selection.h:14`) | a `kernel` is a Selection: `weight_fn` + `params_json`, resolving to `weights` (`selection.ts:224`); it is also an operator factor (a gain, `model.sql:847–856`) and a brush's tip | a kernel is a Selection. A measurement's weights come from its selection's `weights` arm; a spectral gain is a kernel Selection factor of a composed operator |
| A3 | ladders and levels | a constant-Q ladder's levels are tilings; a dyadic grid stores only its finest tile codes and a rule (`params_json.grid`), its levels label and band only; a line's time grid is one `ladder` row with every level closed form and no level rows; a product names a level by `selection_factor.level` (`consolidation.md` §6.2) | a `ladder` is top level with a dictionary; its MEMBERS are the levels, each a stored Selection whose code is read with the ladder's dictionary (`model.sql:565–566`): a constant-Q level a closed-form `spans` (`backend/src/entities/constant-q.ts:53`), a surface level a partition `set` with its own tile array (`backend/src/entities/surface-ladder.ts:1–10`) | levels are member Selections. A dyadic ladder stores each level's array (the codes stay dyadic, so `code >> shift` still rolls up); the time grid's levels are closed-form `spans` members (a few rows); a product's factor is the level's own Selection, and `selection_factor.level` goes |
| A4 | spans | "a set of spans" | `spans` is its own type: rows (events) or closed form (`tiling`); member k is `at(k)` with the derived id `<id>#<k>`; a tiling's ordinal is its code, a row's `code` is its own (an event type) | adopted as the Studio has it |
| A5 | the partition property | a stored family, `grid` or `irregular` (`params_json.family`; `selection.h:61`); the graph's `SelectionShape.partition` flag (`types.h:151`) | nothing stored: a chunk grid is found BY DEFINITION, by testing that a spans set tiles the axis (`field.ts:99–166`); measurements are stored at a tiling's finest level and rolled up by `code >> shift` (`model.sql:617–619`) | the family and the flag go. Whether a selection partitions its manifold is a derived predicate of its type and rows; a roll-up is refused when the selection is not a dyadic tiling or ladder level, read the same way |
| A6 | grid versus irregular; the elements grid | two families of sets; an "elements grid" (every channel its own member, `params_json.grid {kind: elements}`) | no families; per-element values over windows are ARRAY-BACKED measurements whose rows are the field's elements (`[rows × members × components]`, D96; `model.sql:620–624`) | no families; the elements grid goes: a value per channel per member is an array-backed measurement on the member selection, its rows the field's channels |
| A7 | member identity and labels | a set per element stores each element's MEMBER NUMBER, and each member's label in a sibling array `P.labels` (`selection_description.h`; `selection-module.md` §0 item 2) | a partition's array holds each element's CODE, read with the dictionary; its members are "the elements whose code is k" (D48); a measurement's `code` is that code | the array holds codes. Distinct members sharing a label (five bursts of one type) are listed members (`spans` rows, or member `indices` Selections each with its `code`), as the Studio writes them |
| A8 | irregular members by centre and extent | `span_about` (a span from a centre and an extent per axis); a `distance` on a surface | a span is written by its CORNERS; "centre + width is not how a span is written: that is a distance's rule" (`model.sql:596–601`) | `span_about` stays as a maker whose output is a corner span; nothing stores a span's centre and extent |
| A9 | products | factors must be sets; a span or a point is refused (`factor_refusal`); `selection_factor` rows name the factors; a product records its family | a product's record is `params_json.product = [{selection, members}]`; a span or point factor has one member; `kernel`, `indices` and `distance` are not partitions and are refused (`product.ts:26–30`, `:53–73`) | the Studio's rules: a span or point may be a factor; `selection_factor` is dropped (`params_json.product` is the record); no family |
| A10 | the joint partition | `joint(factors)`: time first, its radix rounded up to 2^bits, `code = (tile << bits) \| frame` | the same (`backend/src/db/atlas.ts:11–13`) | unchanged |
| A11 | resolved forms | `Rule`, `Intervals`, `Elements`, `MemberOfElement`, `ProductForm` (`selection-module.md` §8–9) | `resolve()`: crop · indices · weights; `at(k)`, `indexAt`, `factorCodes`, `factorMembers`, `productCode(s)` | the Studio's names and arms (section 5 maps each form); the array kernels behind the forms stay |
| A12 | a position on a space | not in compute's Selection | a `point` on a SPACE: `position` in a coordinate system (the time cursor, D95), snapped to a Line by `vertexOn` | added |
| A13 | a box on a product | a `span` with one extent per factor (`resolve_box`) | the same: an extent's axis is a factor ordinal; on an axis with no closed form the extent is element indices (`selection.ts:187–211`) | unchanged |
| A14 | `detected_over` | a selection's provenance in its shape (`types.h:153`) | the selection row has no producer column | recorded in the producing run's provenance (the commit unit's producer record), not on the class |
| A15 | resolving a distance | `resolve_distance` by the heat method, giving the held elements | the entity does not resolve a `distance` (`selection.ts:225`); the toolbox evaluates it | kept as a compute function whose output is an `indices` Selection |

---

## Appendix B. Open questions

1. **Where the classes live.** The classes are named by compute (functions take and return them), io (it loads and stores them), the graph checker
   (`nxr_graph`, which today links only nlohmann_json) and the bindings. Under the dependency table, io may never include compute. Options: (a) a target
   of its own, `nxr_model`, with no file access and no maths beyond index arithmetic, which compute, io, meeg, the graph and the bindings may all include
   (a dated amendment to the dependency table); (b) the classes in compute, io working in rows and arrays, and the pipelines building the classes from them
   (today's selection bridge, which keeps two descriptions of every class). Recommendation: (a), in its own folder `model/`.
2. **Which list is "the Studio's" where the Studio has several.** The Selection types: the schema's eight (`model.sql:538`), S-CLAUDE §1's six words
   (Span, Point, Labels, Indices, Kernel, SpanSet; lines 34–36, where Span is "centre + extent") and the Studio's input port's four kinds (vertices,
   partition, spans, cells; `ports.ts:256–260`). The ℂ width: 3 embedded (`fieldForms.ts:76–77`, `valueType.ts:12`) or 2 in a frame (`ports.ts:135`,
   `surfaceKit.ts:452–453`). Recommendation: the schema and the entities are the typing (the eight types; a span by its corners; ℂ embedded, 3 a vertex);
   S-CLAUDE §1's words and the port kinds are names for them (Labels = a partition `set`, SpanSet = `spans`), and the 2-in-a-frame ℂ is a port form only.
3. **Values the Studio's vocabulary does not hold.** nxr-neuro makes complex values in TIME (the analytic signal of a band: a phase), complex vector3 (the
   analytic signal of a free current) and the monogenic signal. The Studio's ℂ is a tangent vector in SPACE, and `algebraOf` reads a stored `complex` as ℂ,
   so a typed graph would offer `div` and `curl` of an analytic signal. Options: (a) the algebra stays the value at a cell (ℝ ℂ ℍ) and a separate part
   of the type says the value is complex in time (stored `complex` × `n_components`, its re/im the last axis), proposed to the Studio as an addition;
   (b) an analytic signal is two ℝ Fields (re, im) linked by lineage. Recommendation: (a), agreed with the Studio before step 3.
4. **nxr-neuro's type facts with no place on a class.** The graph types a field's band and spatial level (`Resolution`), its spin and gauge (G5), its
   non-linearity, and a selection's `detected_over`. The Studio has none of them as class members: a band is a `spans` Selection on the frequency line, a
   level a ladder member, the gauge a surface attribute and a ℂ field's frame, non-linearity a lineage check. Recommendation: they stay checker facts,
   derived from the port's lineage and the Selections it references, never class members (they fail the membership test).
5. **Stored data rewritten by "exactly".** Appendix A changes stored selections that schemas 52 and 53 wrote this round: a partition's array from member
   numbers to codes (A7), a dyadic ladder's levels as member rows with their arrays (A3), the time grid's levels as rows (A3), `selection_factor` and
   `params_json.family` dropped (A3, A5, A9), elements grids replaced by array-backed measurements (A6). Recommendation: accept, with a written migration
   run on development data, and the schema version raised.

---

## Checklist (the architecture policy)

- Can each new compute function be called with a matrix from anywhere? Yes: a Field is built from any matrix (4.3); the maths inside stays matrix in,
  matrix out.
- Can the pipeline run the same chain on another input of the same kind just by naming it? Yes: ports are typed by class, and `run`'s find clauses are
  unchanged.
- Is every loop over samples or chunks, and every file access, inside io? Yes: `load`, `create` and the chunk service are io's; the classes hold state only.
- Is every choice of which data made by the caller? Yes: a port declares a class and its type, never a particular row.
- Reductions remain measurements over Selections (2.7, 4.3); the four classes add no reduction route.
