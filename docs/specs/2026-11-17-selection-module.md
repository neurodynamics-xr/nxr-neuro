# The Selection module (spec, 2026-11-17: in discussion)

Status: section 0 is the settled model; the later sections record how it was reached and the forms that implement it (`compute/include/nxr/selection/`,
io's selection store).

## 0. The model (fixed 2026-11-17)

The selection model is the Cortical Flow Studio's (the schema's `selection` table and its detail rows; io's selection entity is its port). A SELECTION is
a named part of a manifold. Its type says how it is described:

| type | described by | examples |
|---|---|---|
| `point` | one element (or a position on a space) | a seed vertex, an instant |
| `span` | bounds: `[start, stop)` on each axis it constrains, in that axis's coordinates; on a product manifold, one extent per factor (a BOX) | a time range, a frequency band, a box in time × channels |
| `distance` | a centre element and a radius under the manifold's metric | a geodesic disc (heat distance); an irregular member on the cortex |
| `indices` | an explicit list of elements | a drawn region, a burst's vertices, picked channels |
| `set` | members given as selections, as a member number per element, by a rule (a tiling, a grid's closed form), or as a product of grids | a time tiling, the dyadic tiles, bursts, the DK atlas, chunks × channels |

1. **Five types**: point, span, distance, indices, set. A `kernel` is not a selection (it is the weights a measurement uses). `spans` and `ladder` are not
   types of their own: a set of spans, and a grid with levels (stored as the schema's `spans` and `ladder` rows; io maps them).
2. **Index, label, lookup** are three things, as in the Studio's atlases and dictionaries:
   - a member's INDEX is its position in its set (its member number, 0 to n − 1): its identity, what a measurement is kept by;
   - its LABEL is a code per member (`code` of a listed member; `member_codes` of a set per element); two members may share one (five bursts, one label);
   - the LOOKUP (the set's dictionary) maps a label to a name and a colour; label 0 is `unassigned` where a dictionary has it.
   A set per element stores each element's member number, never its label; an atlas imported with labels per element is mapped to member numbers once
   (`from_labels`: one member per distinct label, in ascending label order). There is one count: the members.
3. **A PARTITION is a GRID of the MANIFOLD**, defined before any data: chunk grids, frame tilings, constant-Q ladders, dyadic spatial tiles, the elements of
   a points manifold (each channel its own member). Disjoint, covering, regular, navigational: grids are how measurements are addressed and stored. Their
   HIERARCHY IS CLOSED FORM:
   - a dyadic grid's member number is its tile code h·2^D + t at the finest level D (empty tiles are empty members), and its member at level l is that number
     shifted, `>> (D − l)`;
   - a time grid's level k is the tiling of 2^k samples from the first sample (the chunk grid is k = 15); a constant-Q ladder is a list of such tilings;
   - a level's identity is its position among the levels; its label is its power of two (tiles of 2^k samples; 2^l tiles a piece); it carries the band of
     the spectrum it resolves.
   No parent maps, no level rows per member. Roll-ups exist only on grids, and are this arithmetic.
4. **IRREGULAR selections partition the FIELD**: they are made by detection on measurements over a grid (thresholding an amplitude envelope or a band power):
   bursts, events, scouts, and the DK, DKT and Destrieux atlases (a parcellation by a classifier, not a rule). Sparse, they may overlap, their extent varies;
   they have NO levels and no roll-ups; a statistic across their members is a `summary`. They are not indexed by a grid: the grid a family was detected over
   is its provenance (`detected_over`), and a member's grid address is a lookup made on request (`grid.h`: `address_on_line`, `dyadic_address`).
   The workflow is always: measure over grids → detection partitions the measured field → irregular selections, each measured member by member.
5. **Irregular members are described by CENTRE + EXTENT**: on a line, a centre and a duration (`span_about`; [start, stop) derived); on the cortex, a centre
   vertex and an extent radius (a `distance`). The centre lies in some cell of a grid; the extent decides the grid's level: the level is the first whose tile
   is at least the extent, the cell the one holding the centre (the Studio's `supportOf`).
6. **Product manifolds** are the Studio's (D37: manifold type `product`, geometry form `product`, one `geometry_dimension` row per factor; the cells one rank
   per factor, `'1,0'`). Sensor data are channels (points) × time (a line). A transform (an FFT, Morlet wavelets within a band, spectral kernels within an
   eigenvalue band) makes a product space by `apply` of an operator the node defines; its output domain carries the line it makes as a FACTOR.
7. **Product selections** on product manifolds:
   - a BOX is a `span` with one extent per factor (multi-axis), resolved axis by axis (`resolve_box`);
   - a GRID is the product of GRIDS, one per factor (the Studio's D100 `productSelection`): its member code is mixed radix with the FIRST factor fastest,
     `code = c₀ + n₀·(c₁ + n₁·(c₂ + …))` (`product_code`, `product_codes`), the radices fixed when it is made; a factor may be an irregular set (D100's own example:
     the regions of an atlas × the windows of a time level: a measurement layout), and level roll-ups run only along GRID factors (`rollup_refusal`, `coarser_code`);
   - the JOINT partition (the Studio's D129) is such a product with the time factor first and its radix rounded up to a power of two, so
     `code = (tile << bits) | frame` and arrays are laid out `[band × tile × frame]`;
   - levels are chosen per axis, or COUPLED: a frequency band sets the time level as the Studio's `fitLevelOf` does (the dyadic tile that holds 4 cycles of the band's
     CENTRE, `time_level_of_band(lo, hi, sfreq)`), an eigenvalue band sets the dyadic spatial level (Weyl: the level whose band 4π·2^(l−1)/A to 4π·2^l/A holds its centre).
   A single band SELECTS: it is not a frequency axis. Resolving frequency within a band is a transform (item 6), whose made line is then a factor.
8. **Halos are planner facts**: the margin a node reads round a block (a filter's tail, an interpolation's half-width) is declared by the node to the planner.
   It is never a selection.
9. **Rounding** of caller-stated selections (seconds, Hz) onto samples or bins: the sample nearest the centre is always held, the extent rounds outward
   (start down, stop up), never empty by rounding; a radius or extent 0 holds the centre sample. A grid's own cells assign each sample to the one cell its
   coordinate lies in. Detected members' bounds come from the grid they were detected on (whole cells, unchanged by rounding).
10. **`measure` takes any selection**; disjointness is required only for roll-ups and `within` (grids).
11. Compute creates selections in memory (`compute/include/nxr/selection/`: `selection.h` the types and products, `partition.h` the grid methods, `grid.h`
    coupled levels and addresses, `resolved.h` the resolved forms, `scales.h` named spatial scales, `constant_q.h` the band-design arithmetic); io stores,
    loads and updates them (`selection_store.h`); the pipelines bridge the two. A loaded selection follows the same rules as a made one.
12. How a set is STORED (selection_description.h, model.sql; schema 52). Members are file-backed, never a row per member:
    - an irregular set per element: the `set` row's array `P` holds each element's member number, `P.labels` each member's label (absent when member i is
      labelled i), the dictionary one row per label;
    - a tiling (chunks, frames): a `spans` row with its rule, nothing per member; listed spans (bursts, events): `selection_span` rows;
    - a dyadic grid: its member array of tile codes and `params_json.grid` {kind, depth, slots, pieces} (`slots`: the piece slots the codes reserve, at least 2; `pieces`: the connected pieces); a whole grid with levels is a `ladder` row (`save_grid`); an
      elements grid: a `set` row with `params_json.grid` {kind "elements", size}; a constant-Q ladder: the `ladder` row, a level's identity its
      `member_ordinal`, its label (`code`) the power of two;
    - a product: a `set` row, nothing per member, its factors in `selection_factor` (in order) and their radices in `params_json.product`
      [{selection, members}]; a measurement's `code` is the product code (first factor fastest).
    A database made before is converted by `migrate_selection_storage` (io/migrate.h), recorded in meta `selection_storage`.

The sections below record how this was reached; where they use other words (`Partition`, `Spans`, `Regions`, levels with parent maps), section 0 governs.

## 1. Why

The selection model (the schema's `selection`, `selection_element`, `selection_span`, `selection_extent`, `selection_measurement`; the Studio's decisions
D30–D36, D46, D48, D60, D87, D94–D96) already covers what the pipelines do with partitions. The pipelines grew their own words for it (segments, windows,
frames, chunks, tiles, leaves, cells, joint, pyramid, group tree) and their own copies of the rules (a tiling's closed form, the joint code, the dyadic
roll-up). This module gives the model one vocabulary and one implementation per layer, ported from the Cortical Flow Studio, whose geometry (heat
distance, the log map) is geometry-central's, as nxr-neuro's is.

## 2. What a selection is

A named part of a manifold (or a position on a space): which cells are in play. Every selection has: what it lies on (a manifold and which cells, or a
space), a type saying how its members are written, a code read through a dictionary, and what was measured on it. Measurements are not selections: a
measurement is a reduction over the members of ONE partition.

Per type, two jobs: MAKING it (its constructor, from its own arguments) and RESOLVING it (the concrete part of the manifold: index ranges, indices, or
weights).

## 3. Span and distance

- **span**: a rectangular box written by its corners, `[start, stop)` on each axis it constrains, in that axis's coordinate system (`selection_extent`
  rows). Its rank is the number of constrained axes; an axis left out is all of it. A span is a chain of top-rank cells (a line's edges, D94). It resolves
  to index ranges: corners rounded onto samples on an axis with a closed form, taken as element indices on an axis without one (points). A member of a
  `spans` set is a span.
- **distance**: everything within a radius of a centre under the manifold's metric. It needs a metric, not coordinates; its rank is the manifold's. On a
  line a distance is the span `[c - r, c + r)`.

On a surface:
- **distance** is the geodesic disc, by the heat method (geometry-central's `HeatMethodDistanceSolver` on the `IntrinsicSurface`; the Studio's
  `heatMethod.ts`, `geodesic.ts`).
- **span** needs a chart. Default: the log map round a centre (geometry-central's vector heat method), oriented by the canonical frames (north, west): a
  box in those coordinates is a geodesic rectangle. Alternatives: the registered sphere's latitude and longitude (global, distorted); the embedding's
  coordinates (extrinsic).
- **spans** (a tessellation) is a level of the surface tile ladder: equal-area tiles as a line's tiling has equal-duration intervals.

## 4. The grammar

Verbs:

| word | meaning |
|---|---|
| `select` | make a selection of a type from its arguments |
| `at` | restrict a read to a selection (a crop or a gather) |
| `over` | reduce per member of a partition: a measurement |
| `within` | group a measurement by another partition's members |
| `where` | find stored things by their attributes (unchanged) |

`where` chooses WHICH node; `at` chooses WHICH PART of a node. The time range of a run is a span on the recording's time line under `at`
(`--from/--to` its shorthand, `--all` no `at`).

Combinators: `union`, `intersect`, `subtract` (the Studio's SelectOps); `product` (space level x time level: the joint cells); `level(k)` of a ladder;
`member(k)` of a set or spans; `coarser(shift)`, the dyadic roll-up at read (`code >> shift`).

What the old words become:

| old | new |
|---|---|
| segments, windows, frames, chunks, a time line's tiles | `spans` on a line (closed-form tiling) |
| constant-Q ladder, frame tower, surface tile ladder | `ladder` |
| leaves | a level of a surface ladder |
| cells, joint, frame cells | `product` of two partitions |
| pyramid | measurements `over` a ladder level, rolled up at read |
| group tree | a ladder on the template + a correspondence (an operator) from each subject |
| bad segments, bursts, events | `spans` (rows) on a line |
| channel picks, sources, support | `indices` |
| parcellation, flags, channel names | a partition array with a dictionary |

WHERE IT LIVES (2026-11-17; replaces the earlier "its own library"): selections are handled as every compute result is. There is no separate
selection library.

1. CREATING a selection is compute: a method makes it in memory (`compute/include/nxr/selection/`), with the surface's geometry through
   `IntrinsicSurface` (geometry-central).
2. SAVING it, 3. SAVING MEASUREMENTS on it, 4. LOADING it and 5. UPDATING it are io: one selection CRUD (today scattered over the writer, the task API's
   `SelectionOut` and `MeasurementOut`, and the selection entity, to be gathered). A loaded selection follows the same rules as a made one.
6. The pipeline runner is the bridge (a node makes it, the runner stores it), and the grammar below is carried out across the layers: `select` and the
   set combinators by compute, `at`, `level`, `member` and `coarser` by io at read, `over` and `within` by compute measurement nodes stored by io.

The few rules of the stored format (the tile of a sample, a coarser level as code >> shift, a closed tiling's members) exist in both, as io knows a
field's shape without compute. The selection CRUD is gathered in io's selection module (`selection_description.h`, `selection_store.h`, `selection_measurement.h`; the entity `selection.h` resolves a stored selection); io's constant-Q writer no longer designs its bands: compute designs the ladder (`nxr::selection::constant_q`) and the pipelines hand it to io, which only stores it (`pipelines/constant_q.h`).

A PARTITION OF A MANIFOLD BY A METHOD (`compute/include/nxr/selection/partition.h`): one function, `partition(manifold, method)`; the manifold
says what is divided (a Line's samples, a Surface's vertices), the method how. A partition made here and one io loads from file are the same thing and
follow the same rules. Levels are named by their power of two (a line: tiles of 2^k samples; a surface: 2^l tiles a connected piece), never by rank; the
member of an element at a level is the closed form (a surface: the tile code >> (D − l); a line: element >> k); each level carries the band of the
manifold's spectrum it resolves.

| manifold | method | what it gives |
|---|---|---|
| Line | constant-Q | the levels a constant-Q band design needs (the chunk grid is the same partition at its own k), each with its band in Hz |
| Surface | dyadic | equal-area geodesic bisection (the Studio's ladder; Dijkstra inside the bisection, as there), each level paired with its eigenvalue band by Weyl's law |
| Surface | transfer | another surface's partition carried through the registered spheres, nearest vertex piece by piece (the group tiles are this, from the template's dyadic partition) |
| Surface | rings, polar, poles, cells, nodal | to come: the Studio's coordinate grids (heat distance, the log map, farthest-point Voronoi, band-filtered nodal domains) |

THE SELECTIONS COMPUTE CREATES (in memory; `compute/include/nxr/selection/`): two types, each with an optional LOOKUP (a code's name and colour; io
stores it as the selection's dictionary).

- `Partition`: every element in one member, at nested levels; members by arithmetic (a time line), by member number per element (a surface), or by edges (a line divided at
  boundaries). Made by a method (constant-Q, dyadic, transfer), or by `from_labels` (labels per element and a lookup, mapped to member numbers: a parcellation, flags). The bands a
  constant-Q or dyadic partition's levels resolve partition the spectrum's own line: `spectrum_partition` gives the FREQUENCY partition of a constant-Q one
  and the EIGENVALUE partition of a dyadic one.
- `Spans` (labelled spans): intervals [start, stop) on a line with a code each, which may leave gaps and overlap: events on the time line (bursts,
  spindles, bad segments), named bands on the frequency line (the physiological bands), ranges of eigenvalues.

SELECTIONS THAT ARE NOT PARTITIONS (they may leave elements out and overlap), each labelled through a lookup:
- on a line, labelled `Spans` (`spans.h`): bursts, events from Brainstorm (a span, or an INSTANT where start = stop), bad segments, a span drawn by
  hand; one member of a line's partition as a span (`member_span`: a time tile, a band of a spectrum's partition).
- on a surface (or any manifold of elements, e.g. channels), labelled `Regions` (`regions.h`): sets of elements each with a code. Made by `disc` (heat
  distance through the chart kit), `member_region` (an atlas region, one tile), or from the elements a detector or a drawing gives (a spatial burst's
  region, a region drawn by hand).
Events that are both (a burst on some channels for some time) are the joint case, to come.

NAMED SPATIAL SCALES (`scales.h`): the surface's counterpart of named frequency bands. Hemisphere, lobe, region, gyrus, … are labelled spans on the
eigenvalue line, made from the caller's ranges either by WAVELENGTH (lambda = (2 pi / w)^2) or by the NUMBER OF PARTS of a connected piece (lambda =
4 pi m / A, Weyl's law, as the dyadic levels; the counts may be read from an atlas). The ranges and names are never defaulted.

The same logic holds on every line: time, frequency, eigenvalue.

A subject's own atlases (DK, DKT, Destrieux) are FreeSurfer's (its registration and trained classifier); they are imported with the anatomy, not computed.

## 5. Examples (illustrative)

```bash
nxr-neuro select span      --on rest.time --from 120 --to 180 --name eyes_closed
nxr-neuro select distance  --on cortex --at stim_site --radius 10mm --name disc10
nxr-neuro select span      --on cortex --at stim_site --north -10:10 --west -5:5 --name strip
nxr-neuro select spans     --on rest.time --width 2s --name epochs_2s
nxr-neuro select ladder    --on cortex --dyadic --depth 8 --name tiles
nxr-neuro select product   tiles:L4 cq:L3 --name tiles4_x_cq3
nxr-neuro select subtract  eyes_closed bad_segments --name clean_eyes_closed

nxr-neuro run psd --recording rest --at eyes_closed
nxr-neuro measure power of rest_kernel_current over tiles4_x_cq3
nxr-neuro measure mean of rest_alpha_power over epochs_2s within dk
nxr-neuro measure read power of rest_kernel_current over tiles:L2      # rolled up at read
```

```python
nn.select.distance(on="cortex", at="stim_site", radius="10mm", name="disc10")
nn.measure("power", of="rest_kernel_current", over=nn.select.product("tiles:L4", "cq:L3"), within="dk")
```

## 6. Open questions

1. `set`, `spans` and labelled sets (a parcellation): one partition concept with encodings, or types as now.
2. Is a kernel a selection, or a weight field (a soft membership) used by measurements.
3. Is a ladder a type, or a hierarchical partition whose coarser levels are views (`coarser(shift)`).
4. Spatiotemporal selections: products, and events that move.
5. `select` a top-level verb or a family of definitions; `at` a YAML key on any input; the measurement phrase's keyword.

## 7. Inventory: every compute takes a selection (2026-11-17)

An inventory of compute's public functions (`compute/include/nxr/`, from the headers' signatures and doc comments) found that every compute works over a
selection. What differs is the selection's shape and what the compute returns:

| the selection | its members | the compute returns |
|---|---|---|
| single elements | each element alone | a value per element (pointwise) |
| a neighbourhood per element | the span ±h around a sample, a vertex's one-ring, a disc (overlapping, one per element) | a value per element |
| a partition | disjoint windows or regions | a value per member |
| labelled spans or regions | events, bursts, discs (may overlap) | a value per member |
| the whole manifold | the whole recording or connected piece | a value for the whole |

Today the selection is passed in disguise, as a plain parameter under a name of its own. This list is the refactoring work: each becomes a selection
resolved into one fixed array form, and the diverging words (tiles, windows, segments, frames, pyramid, groups, nodes, cells) dissolve into the selection
vocabulary.

### 7.1 Reductions within members (a selection required)

| how the selection is passed today | functions | what it is |
|---|---|---|
| a label or member number per element and a count, under five names (`codes`, `node`, `rowGroup`, `members`, `Surface.node`) | `partition::accumulate`, `medians`, `meanOverMembers`; `ladder::scalarSummary`, `frameSummary`, `vectorSummary`, `leviCivita`, `lcResultant`, `tileAreas`, `tileCentres`; `integrals::flux`, `circulation`; `tileFlux`, `tileCirculation`; `region::collapse`; `quadrature::groupedCongruenceNorms`, `pairCongruenceNorms`; `crossings::crossings` | a partition by codes |
| a tile width and a tile count | `partition::reduceTiles`, `crossSpectrum`; `histogram::histogramTiles` | a time partition at one level |
| a segment length and an overlap | `welch`, `Welch`, `ComplexWelch`, `LinearWelch`, `CrossWelch` | a regular family of time spans (overlapping segments) |
| a tile index per sample (`tile[samples]`) with codes per row | `partition::analyticSums`, `tensorSums`; `pyramid::bandSums`, `tensorBandSums`, `frameOfSamples`, `addSampleCounts`; `tower::sampleCounts` | a product partition (space × time) |
| bin groups `{first, last}` | `CrossWelch` | a frequency partition |
| span and event lists (`TileEvent`, `Span`), an event-label matrix (`Labels`), a span as two numbers | `intervals::rollupEvents`, `eventDurations`, `eventIntervalNext`; `extent::rollup`, `areaByFrame`, `centres`, `pathLength`, `windowRegions`; `risefall::fit` | labelled spans; events on a product manifold |
| a baseline per member as numbers (`median`, `sd`) | `extent::standardize` | a measurement over a selection, handed in |

### 7.2 Reductions across members

`stats.h` (mean, sd, median, mad, percentile, peak, skewness, kurtosis, … of a vector of member values); `histogram::sumTiles`, `percentile`, `median`,
`mad` (a run of tiles as `first, last`); `partition::sumOverMembers`; `tower::rollTime` (a coarser level as a merge); `pca::pca` (members as rows);
`scale::selectScale`, `eventFacts`; `scalespace::selectLevel` (across levels); `surrogate::besagCliffordP`, `benjaminiHochberg`; `crossings::repeat`,
`poolSenses`. They take the members' values already assembled by the caller.

### 7.3 Neighbourhoods

| how the neighbourhood is given today | functions |
|---|---|
| derived from a design, never stated (the streaming state handles chunk edges) | FIR filters (length from ripple and transition width); IIR notch and bandstop (unbounded, forward and back); `Filter::window`, `RunningMean` |
| an explicit margin or halo, computed and reported | `wt` margin (the longest member's response tail); `timebank::haloOf`; `frame_tiling::sincInterpolate` (`half_width`); `taps` (half-width); `surrogate::shapingTaps`, `filterValid` |
| the mesh's one-ring, implicit in the operator (k-rings for a Chebyshev series of order k) | the Laplacians, DEC, Hodge–Dirac and connection operators; curvatures, normals, face frames; `phase::vertexGradient`, `phaseGradient`, `phaseVelocity`, `travellingIndex`; `hodge_weights`, `intrinsic_forms`, `magnetic`, `transport`, `lag::holonomy`; the graph filterbank's Chebyshev route |
| a radius or a distance | `ChartKit::disc`; `quadrature::weightedPhaseGradient` (a disc); `tracks::link` (`linkDistance`); `lag::pairsAtDistance` |
| the whole connected piece, or the whole block passed in | eigensolvers, Poisson, heat distance, vector heat, Hodge decomposition, the filterbank's exact route, `jointbank`, `spacetime_kernels::scatter`, `monogenic`, optical flow; the FFT Hilbert transforms of `opticalflow::hilbertRows`, `activation` and `pyramid::spectrumOf`, `analyticBand` |

Where the neighbourhood is "the whole block passed in", the result depends on where the pipeline cuts its chunks: the chunking decides the neighbourhood,
not the method. An explicit neighbourhood selection makes this visible, and tells the planner the margin or the whole member to read.

### 7.4 Pointwise (the selection is each element)

`analytic::abs`, `angle`; `rotary`; `scaling`; `phase::power`, `phaseCurrent`, `gradientPower`; `spectral_scaling::convert`, `linearFactor`, `unitOf`;
`congruence` (A M Aᴴ per sample or pair); `quadrature::cospectrum`, `quadspectrum`, `phaseLag`, `timeLag`, `circ_r`;
`pyramid::bandGains`; `lag::lagStatistics`, `circularShift`; `surrogate::normalAt`, `whiteBlock`, `wishartDraw`.

### 7.5 Selections already made in compute, not recognised as such

They create selections and belong with the selection makers: `tower::makeFrameTiling`, `frameSpan`, `parent`, `levelForSeconds` (time frames);
`regions::detectRegions`, `cores::corePatches`, `extent::windowRegions` (spatial regions and events); `intervals` event detection (spans);
`lag::pairsAtDistance` (pairs of vertices). The `good` sample mask passed to Welch, the pyramid and several reductions is itself a selection (the
complement of the bad segments).

### 7.6 Not classified with confidence

`speedspectrum::compute`, `pyramid::frameKernel`, `quadrature::rotarySpectrum` (a window per frequency), `connectome::partitionCounts`,
`smoothMemberships`, and compute.h's generators, loci and isolines; the covariant operators and vector-field operations' neighbourhoods were not read at
line level.

## 8. Resolved forms: the one array form each compute takes (2026-11-17)

A selection says what a part of a manifold IS; a compute needs which elements it holds, as arrays it can loop over. Section 7 found the same thing passed
under many names. They become FOUR forms, one name each (`compute/include/nxr/selection/resolved.h`, `compute/src/selection_resolved.cpp`), resolved from
a `Selection` by one resolver each, plus the product and box forms of section 9:

| form | what it holds | arrays | resolver | accepts |
|---|---|---|---|---|
| `Rule` (the closed form) | a regular family of intervals on a line, by a rule, never materialized: member k = [origin + k·hop, origin + k·hop + width) ∩ [0, size) | origin, width, hop, count (in points), the line's size | `resolve_rule(s, line)` | a tiling whose origin, width and hop are whole points of the line |
| `Intervals` | per member, sorted disjoint [start, stop) point intervals on a line | CSR: `offsets` [members + 1], `start`, `stop` | `resolve_intervals(s, line)` | point, span (one axis), distance (on a line), indices, and sets of these: listed, a tiling, a set per element over the line's points, an elements grid |
| `Elements` | per member, its sorted elements (members may overlap) | CSR: `offsets` [members + 1], `elements` | `resolve_elements(s, size)` | point, indices, a set per element (irregular, or a dyadic grid by tile code), an elements grid, a set of listed points and indices |
| `MemberOfElement` | the member NUMBER of each element (−1: in none) and the member count; DISJOINT members only | `member_of` [size], `members` | `resolve_member_of(s, size)`, `resolve_member_of(s, line)` | as `Elements` (or `Intervals` on a line), with disjoint members |

Rules shared by every form:
- **A member's identity is its member number, and every form keeps every member**, an empty one included (a span outside the line, a tile no vertex reaches):
  a measurement's k-th value is always member k's. An irregular set per element is numbered by its own dense member numbers; a dyadic grid by its tile codes
  (slots·2^D of them); an elements grid by its elements.
- **Rounding onto a line** (section 0, item 9):
  - a grid's cell [a, b) holds the points whose coordinate lies in it, [⌈(a − origin)/spacing⌉, ⌈(b − origin)/spacing⌉), so each point is in one cell;
  - a caller-stated extent (a span, an irregular member, a band in Hz) rounds OUTWARD, [⌊(a − origin)/spacing⌋, ⌈(b − origin)/spacing⌉), and always holds
    the point nearest its centre: never empty (an instant is the one point nearest it);
  - a distance on a line, centre element c and radius r, holds [c − h, c + h] with h = ⌈r/spacing⌉ (a radius 0 holds the centre);
  - each with a 1e-9 tolerance, so a corner on a point stays there; a unit on the selection must be the line's.
- **A tiling resolves to a `Rule`** only when it is whole points of the line; otherwise `resolve_intervals` assigns its points cell by cell. The rule's origin
  may lie before the line (a block that starts inside a tiling): `rule_member_of(rule, first, n)` gives the member of each point of a block without the whole
  line's member numbers, `rule_members_at(rule, i)` the members holding a point (several when hop < width: Welch's overlapping segments, a family that is not a
  partition), `rule_member(rule, k)` one member's interval.
- **Refusals name the way forward**: a grid with levels (name one: `level(set, k)`, or the level a band sets, `grid.h`), a product (`resolve_product`), a
  span on several axes (`resolve_box`), a span or a tiling asked for as elements (they lie on a line: `resolve_intervals`, then `elements_of`), a distance on
  a surface (it needs the metric: `resolve_distance` first, which always holds the centre), members that share an element asked for as a member number per
  element.
- **Conversions** where exact: `intervals_of` (Rule, Elements, MemberOfElement), `elements_of` (Intervals, MemberOfElement), `member_of_from` (Elements,
  Intervals, Rule; disjoint members only). `resolve(s, Space)` picks the resolver from the form a caller names (`FormKind`) and where the selection lies.
- Indices are 64-bit on a line and in `Elements` (hours of samples; a product manifold's cells); member numbers are 32-bit.

What each disguised parameter of section 7 becomes (the later waves rewrite these functions to take the form):

| today (section 7) | functions | the form |
|---|---|---|
| `codes`, `node`, `rowGroup`, `members`, `Surface.node` + a count | `partition::accumulate`, `medians`, `meanOverMembers`; `ladder::*Summary`, `leviCivita`, `lcResultant`, `tileAreas`, `tileCentres`; `integrals::flux`, `circulation`; `tileFlux`, `tileCirculation`; `region::collapse`; `quadrature::groupedCongruenceNorms`, `pairCongruenceNorms`; `crossings::crossings` | `MemberOfElement` |
| a tile width and a tile count | `partition::reduceTiles`, `crossSpectrum`; `histogram::histogramTiles` | `Rule` (hop = width) |
| a segment length and an overlap | `welch`, `Welch`, `ComplexWelch`, `LinearWelch`, `CrossWelch` (and `segmentFor`, which makes the tiling) | `Rule` (hop = n − ⌊n·overlap⌋; members overlap: a family, not a partition) |
| a tile index per sample (`tile[samples]`) with codes per row | `partition::analyticSums`, `tensorSums`; `pyramid::bandSums`, `tensorBandSums`, `frameOfSamples`, `addSampleCounts`; `tower::sampleCounts` | `ProductForm` of (`Rule` on time, `MemberOfElement` on the rows), time first (section 9); `rule_member_of` gives a block's tile per sample |
| bin groups `{first, last}` (inclusive) | `CrossWelch`; `bandAmplitude`'s `BinRange` | `Intervals` on the frequency line (bins: a `Line` of count bins, spacing Δf, in Hz), rounded outward |
| span and event lists (`intervals::Span`, `TileEvent`), a span as two numbers | `intervals::rollupEvents`, `eventDurations`, `eventIntervalNext`; `extent::rollup`, `centres`, `pathLength`; `risefall::fit` | `Intervals` (one member per event, each described by centre + extent: `span_about`); a `TileEvent` is an interval of a level's tiles. Their roll-up across members is a summary statistic, not a grid roll-up |
| an event-label matrix (`Labels`), events of cells on tiles × time tiles | `extent::areaByFrame`, `windowRegions` | `MemberOfElement` on the product manifold's cells, or `Elements` when events overlap (section 9): irregular, not a product |
| the `good` sample mask | Welch, the pyramid, several reductions | `Intervals`, one member: the good samples (the complement of the bad segments, `subtract`) |
| a baseline per member as numbers | `extent::standardize` | stays numbers: a measurement over a selection, handed in (not a form) |
| a neighbourhood: a half-width, a margin, a halo (section 7.3) | `timebank::haloOf`, `frame_tiling::sincInterpolate`, `taps`, `surrogate::shapingTaps`, `filterValid`, `wt`'s margin | NOT a selection: a planner fact the node declares (the margin it reads round a block); the function keeps its half-width parameter |
| a radius or a distance | `ChartKit::disc`, `quadrature::weightedPhaseGradient`, `tracks::link`, `lag::pairsAtDistance` | a `distance` selection, resolved by `resolve_distance` to `Elements` |
| selections made in compute (section 7.5) | `tower::makeFrameTiling`, `frameSpan`, `parent`; `regions::detectRegions`, `cores::corePatches`; `intervals` detection; `lag::pairsAtDistance` | a frame tiling is a grid (`tiled`, `time_grid`; `parent` is the closed form `>> 1`); detections return irregular sets (spans by centre + extent, sets of indices); the pair list is a set of pairs, not a product |

## 9. Product manifolds and product selections (2026-11-17)

A measurement per chunk per channel, per time tile per cortical tile, or per frame per tile per band, lies on a PRODUCT manifold (the Studio's D37: manifold
type `product`, geometry form `product`, one `geometry_dimension` per factor naming the factor's manifold; the cells one rank per factor, `'1,0'`). A recording
is on channels (points) × time (a line). A transform makes a product space by `apply` of an operator the node defines (an FFT, Morlet wavelets within a band,
spectral kernels within an eigenvalue band); its output domain carries the line it makes as a factor.

Selections on a product manifold stay within the five types:

- **A box** is a `span` with one extent per factor it constrains (an axis left out is all of it): `resolve_box(span, lines)` gives one interval per axis,
  each rounded outward (a points axis such as the channels is a `Line` of its elements, spacing 1 from 0). A burst on channels 3 to 7 ([3, 8): centre 5.5, extent 5)
  for 200 ms around t is `span_about({{0, 5.5, 5}, {1, t, 0.2}})` on channels × time: one member, not a product.
- **A grid** is `product(factors)` (`selection.h`; the Studio's D100 `productSelection`): a `set` whose factors are sets, one per factor manifold (a tiling, a
  level of a dyadic grid or of a constant-Q ladder, an elements grid, and as D100 has it an irregular set, e.g. an atlas: a measurement layout). A single span or point
  is refused with the reason and what to use instead (`factor_refusal`); the product records its FAMILY (grid when every factor is, else irregular); a grid with levels must name one (or take the level a band sets). A product of products is flattened.
- **Code: mixed radix, the FIRST factor fastest**, `code = c₀ + n₀·(c₁ + n₁·(c₂ + …))` (`product_code(radices, codes)`, `product_codes(radices, code)`; −1
  when a factor's member is none). The radices are fixed when the product is made (each factor's count). So the members of one window are contiguous, and a
  factor's member is recovered by division. An array laid out [factor n−1 × … × factor 0] (row-major, the first factor innermost) is in code order.
- **The joint partition** (the Studio's D129) is `joint(factors)`: a product with the time factor (frames) first and its radix rounded up to 2^bits, so a code
  is `(tile << bits) | frame` and the arrays are [band × tile × frame]. The codes between the frame count and 2^bits are empty members.
- **Coupled levels** (`grid.h`). A single band selects; it sets the level of the other factors' grids:
  - time: `time_level_of_band(lo, hi, sfreq)` replicates the Studio's `fitLevelOf` (jointTime.ts): the support is 4 cycles of the band's centre sqrt(lo·hi), the level
    the first dyadic tile (2^k samples) at least that (`natural_level`), held inside the line, so the time grid is `time_grid(line, k)`; `support_of` is the Studio's
    `supportOf` (the tile holding a time, at the fitted or a coarser level);
  - space: `space_level_of_band(λ_lo, λ_hi, A, D)` is the dyadic level whose Weyl band [4π·2^(l−1)/A, 4π·2^l/A) holds the band's centre (`weyl_level`), so the
    spatial grid is `level(tiles, l)`.
  Resolving frequency (or eigenvalue) within the band is a transform whose made line is then a factor.
- **Label**: the tuple of the factors' labels, each read with its own factor's lookup (`factor_labels`): a grid's member has none (its number names it). A
  product carries no lookup of its own.
- **Partition**: a product of grids is disjoint and covers the product manifold; roll-ups apply to one GRID factor's member (unpack with `product_codes`, shift that
  factor by its closed form, repack with the coarser radices: `coarser_code`, `rolled_radices`; io's `coarser_product_code` reads `params.product` for it). A roll-up along an
  irregular factor, or over an irregular set, is refused (`rollup_refusal`; the graph checker's `summarize` and `reduce_cells` with a `level`).
- **Family**: every set records GRID or IRREGULAR (compute `Selection::family`; stored `params_json.family`; a row from before the mark says what its type and params say).
- **Resolved form: `ProductForm`** (`resolve_product(p, spaces)`, one `Space` per factor saying where it lies and which form to take): the factors' forms (a
  `Rule` for the chunks, `MemberOfElement` for the channels or tiles) with `counts`, `radices`, `codes(k)` and `code(…)`. A compute loops over the factors'
  forms and writes value k at the product code; the product is never materialized.
- **Irregular shapes on a product manifold** (a burst whose channels change over time, a region that moves) are not products: they are `indices` (or an
  irregular set per element) on the product manifold's cells, numbered as the field's array is; `Elements` when they overlap. Their grid address, on request,
  is the cell holding the centre at the level the extent sets (`address_on_line`, `dyadic_address`).
- **Irregular members** (centre + extent) are found from the grid: a time member's level is `natural_level(extent)` and its cell the tile holding its centre
  sample (the Studio's `supportOf`); a cortical member's level is Rheome's `jointcell` rule, `cortical_level_of_extent(sigma, diameters)` (the COARSEST level whose MEASURED tile diameter, the ladder's,
  is at most `tile_sigma` × sigma; a refusal with the reason when none is fine enough) and its cell the tile holding its centre vertex.

Examples (compute):

```cpp
Selection frames = time_grid(time_line, time_level_of_band(8, 16, sfreq));                 // the time grid alpha sets
Selection tiles  = level(dyadic(surface, {}), space_level_of_band(lambda_lo, lambda_hi, piece_area, depth));
Selection cells  = joint({frames, tiles});                                              // code = (tile << bits) | frame
Selection sens   = product({time_grid(time_line, 15), elements_grid(n_channels)});      // chunks x channels: code = chunk + n_chunks * channel
ProductForm f = resolve_product(sens, {{FormKind::rule, time_line}, {FormKind::member_of, std::nullopt, n_channels}});
```

**How io stores a product** (selection_store.h `save` with `factors`, `radices` and `Placement::factor_ids`; schema 52 `selection_factor`):
- One `selection` row: `type = 'set'`, `manifold_id` the product manifold, `cell` the factors' cells joined by ',' (`'1,0'`: time edges × channel
  vertices), `n_members` the product of the radices, `array_path` null (nothing per member is stored), no `dictionary_id` (the factors carry theirs),
  `params_json.product` = [{selection, members}] (each factor's id and radix, as the Studio's D100 writes it).
- Each factor is a stored GRID in its own right, on its own manifold, with its own dictionary; the store refuses an irregular factor (listed spans, an
  irregular set per element, a ladder whose level is not named). The product names them in order in `selection_factor (selection_id, ordinal,
  factor_selection_id)`, primary key (selection_id, ordinal).
- Measurements: `selection_measurement.code` is the product CODE (first factor fastest); an array-backed measurement is `[rows × members]` with the members in
  code order. "Every chunk in channel 3" and `within` one factor's member are arithmetic on the code (division by the radices before that factor, modulo its
  own), which io's measurement query does from `params_json.product`; the roll-up `coarser(shift)` applies to one factor's member (unpack, shift, repack).
