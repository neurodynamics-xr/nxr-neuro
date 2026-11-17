# The Selection module (spec, 2026-11-17: in discussion)

Status: the vocabulary and grammar below are agreed in direction; the open questions at the end are being decided. Nothing here is built yet beyond the io
selection entity (`io/include/nxr/io/selection.h`, a port of the Studio's `entities/selection.ts`).

## 0. The vocabulary (fixed 2026-11-17)

The selection model is the Cortical Flow Studio's (the schema's `selection` table and its detail rows; io's selection entity is its port). A SELECTION is
a named part of a manifold, with a code (its label) read through a dictionary (its lookup). Its type says how it is described:

| type | described by | examples |
|---|---|---|
| `point` | one element (or a position on a space) | a seed vertex, an instant |
| `span` | bounds: `[start, stop)` on each axis it constrains, in that axis's coordinates | a time range, a frequency band, a box in a chart |
| `distance` | a centre element and a radius under the manifold's metric | a geodesic disc (heat distance) |
| `indices` | an explicit list of elements | a drawn region, a burst's vertices, picked channels |
| `set` | members given as selections, as a code per element, or by a rule (a tiling: origin, width, hop, count) | events, bursts, bad segments and named bands (a set of spans); DK (by codes); a time tiling (by a rule); a set of regions |

Changes to the Studio's model, decided here:

1. A `kernel` is not a selection: it is the weights a measurement uses (applying a wavelet to a field is a measurement). Choosing atoms or bands is a
   selection.
2. A `ladder` is not a kind of its own: it is a `set` with LEVELS, a grid hierarchy of nested partitions each refining the one above, named by their power
   of two. `spans` is not a kind of its own either: it is a `set` whose members are spans (listed, or by a rule). (Stored as the schema's `ladder` and
   `spans` rows, unchanged; io maps them.)
3. PARTITION is a property: a selection whose members are disjoint and cover the manifold. Measuring over members needs no property; grouping by members
   (`within`) and rolling up across them need disjoint members.
4. A member's IDENTITY (its ordinal) and its LABEL (its code) are distinct: five bursts are five members with one label. Measurements are kept per member.
5. The same model holds on every line: time, frequency, eigenvalue (spectrum partitions, named bands and scales).
6. Compute creates selections in memory, mirroring these types (`compute/include/nxr/selection/selection.h`: one `Selection`; the partition methods in
   `partition.h`: `constant_q`, `constant_q_bands`, `dyadic`, `eigenvalue_bands`, `transfer`; named scales in `scales.h`); io stores, loads and updates
   them; a loaded selection follows the same rules.
7. A compute takes a selection in one of four RESOLVED FORMS (`resolved.h`: `Rule`, `Intervals`, `Elements`, `Codes`; section 8). A PRODUCT of
   selections is a `set` whose members are given by factors, numbered row-major, labelled by the factors' own lookups (section 9); no sixth type.

The sections below record how this was reached; where they use other words (`Partition`, `Spans`, `Regions`), section 0 governs.

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
member of an element at a level is arithmetic (element >> k; codes[element] >> (depth - l)); each level carries the band of the manifold's spectrum it
resolves.

| manifold | method | what it gives |
|---|---|---|
| Line | constant-Q | the levels a constant-Q band design needs (the chunk grid is the same partition at its own k), each with its band in Hz |
| Surface | dyadic | equal-area geodesic bisection (the Studio's ladder; Dijkstra inside the bisection, as there), each level paired with its eigenvalue band by Weyl's law |
| Surface | transfer | another surface's partition carried through the registered spheres, nearest vertex piece by piece (the group tiles are this, from the template's dyadic partition) |
| Surface | rings, polar, poles, cells, nodal | to come: the Studio's coordinate grids (heat distance, the log map, farthest-point Voronoi, band-filtered nodal domains) |

THE SELECTIONS COMPUTE CREATES (in memory; `compute/include/nxr/selection/`): two types, each with an optional LOOKUP (a code's name and colour; io
stores it as the selection's dictionary).

- `Partition`: every element in one member, at nested levels; members by arithmetic (a time line), by codes (a surface), or by edges (a line divided at
  boundaries). Made by a method (constant-Q, dyadic, transfer), or by `labels` (a code per element and a lookup: a parcellation, flags). The bands a
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
| a code per element and a count, under five names (`codes`, `node`, `rowGroup`, `members`, `Surface.node`) | `partition::accumulate`, `medians`, `meanOverMembers`; `ladder::scalarSummary`, `frameSummary`, `vectorSummary`, `leviCivita`, `lcResultant`, `tileAreas`, `tileCentres`; `integrals::flux`, `circulation`; `tileFlux`, `tileCirculation`; `region::collapse`; `quadrature::groupedCongruenceNorms`, `pairCongruenceNorms`; `crossings::crossings` | a partition by codes |
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
`congruence` (A M Aᴴ per sample or pair); `quadrature::coSpectrum`, `quadratureSpectrum`, `phaseLag`, `timeLag`, `meanResultantLength`;
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
a `Selection` by one resolver each:

| form | what it holds | arrays | resolver | accepts |
|---|---|---|---|---|
| `Rule` (the closed form) | a regular family of intervals on a line, by a rule, never materialized: member k = [origin + k·hop, origin + k·hop + width) ∩ [0, size) | origin, width, hop, count (in points), the line's size | `resolve_rule(s, line)` | a set by a rule (a tiling) whose origin, width and hop are whole points of the line |
| `Intervals` | per member, sorted disjoint [start, stop) point intervals on a line | CSR: `offsets` [members + 1], `start`, `stop` | `resolve_intervals(s, line)` | point, span (one axis), distance (on a line), indices, and sets of these: listed, by a tiling, by codes over the line's points |
| `Elements` | per member, its sorted elements (members may overlap) | CSR: `offsets` [members + 1], `elements` | `resolve_elements(s, size)` | point, indices, a set by codes, a set of listed points and indices |
| `Codes` | a member code per element (-1: in none) and the member count; DISJOINT members only | `codes` [size], `members` | `resolve_codes(s, size)`, `resolve_codes(s, line)` | as `Elements` (or `Intervals` on a line), with disjoint members |

Rules shared by every form:
- **A member's identity is its ordinal, and every form keeps every ordinal**, an empty member included (a span that falls outside the line, a code that
  holds no element): a measurement's k-th value is always member k's. A set by codes numbers its members by the codes themselves (`ordinal_count`: the
  largest code in its codes or its lookup, + 1).
- **Onto a line**: a span [a, b) holds the points whose coordinate lies in it, indices [⌈(a − origin)/spacing⌉, ⌈(b − origin)/spacing⌉) (with a 1e-9
  tolerance, so a corner on a point is that point; the rule `measure.band_amplitude` uses for a band's bins). An instant (a = b) holds the point nearest it.
  A distance on a line is the span [c − r, c + r). A unit on the selection must be the line's.
- **A tiling resolves to a `Rule`** only when it is whole points of the line; otherwise `resolve_intervals` rounds its members one by one, by the span rule.
  The rule's origin may lie before the line (a block that starts inside a tiling): `rule_codes(rule, first, n)` gives the member of each point of a block
  without the whole line's codes (today's `tile[samples]`, `frameOfSamples`), `rule_members_at(rule, i)` the members holding a point (several when
  hop < width: Welch's overlapping segments), `rule_member(rule, k)` one member's interval.
- **Refusals name the way forward**: a set with levels (name one: `level(set, k)`, the grammar's `level(k)`, added to `selection.h`), a product
  (`resolve_product`, section 9), a span or a tiling asked for as elements (they lie on a line: `resolve_intervals`, then `elements_of`), a distance on a
  surface (it needs the metric: `resolve_distance` first), members that share an element asked for as codes.
- **Conversions** where exact: `intervals_of` (Rule, Elements, Codes), `elements_of` (Intervals, Codes), `codes_of` (Elements, Intervals, Rule; disjoint
  members only). `resolve(s, Space)` picks the resolver from the form a caller names (`FormKind`) and where the selection lies (a `Line`, or a size).
- Indices are 64-bit on a line and in `Elements` (hours of samples; a product manifold's cells); codes are 32-bit, as every compute that takes codes reads them.

What each disguised parameter of section 7 becomes (the later wave rewrites these functions to take the form):

| today (section 7) | functions | the form |
|---|---|---|
| `codes`, `node`, `rowGroup`, `members`, `Surface.node` + a count | `partition::accumulate`, `medians`, `meanOverMembers`; `ladder::*Summary`, `leviCivita`, `lcResultant`, `tileAreas`, `tileCentres`; `integrals::flux`, `circulation`; `tileFlux`, `tileCirculation`; `region::collapse`; `quadrature::groupedCongruenceNorms`, `pairCongruenceNorms`; `crossings::crossings` | `Codes` |
| a tile width and a tile count | `partition::reduceTiles`, `crossSpectrum`; `histogram::histogramTiles` | `Rule` (hop = width) |
| a segment length and an overlap | `welch`, `Welch`, `ComplexWelch`, `LinearWelch`, `CrossWelch` (and `segmentFor`, which makes the tiling) | `Rule` (hop = n − ⌊n·overlap⌋; members overlap) |
| a tile index per sample (`tile[samples]`) with codes per row | `partition::analyticSums`, `tensorSums`; `pyramid::bandSums`, `tensorBandSums`, `frameOfSamples`, `addSampleCounts`; `tower::sampleCounts` | `ProductForm` of (`Rule` on time, `Codes` on the rows) (section 9); `rule_codes` gives a block's tile per sample |
| bin groups `{first, last}` (inclusive) | `CrossWelch`; `bandAmplitude`'s `BinRange` | `Intervals` on the frequency line (bins: a `Line` of count bins, spacing Δf, in Hz) |
| span and event lists (`intervals::Span`, `TileEvent`), a span as two numbers | `intervals::rollupEvents`, `eventDurations`, `eventIntervalNext`; `extent::rollup`, `centres`, `pathLength`; `risefall::fit` | `Intervals` (one member per event; a `TileEvent` is an interval of a level's tiles: `Intervals` on that level's tile line) |
| an event-label matrix (`Labels`), events of cells on tiles × time tiles | `extent::areaByFrame`, `windowRegions` | `Codes` on the product manifold's cells, or `Elements` when events overlap (section 9) |
| the `good` sample mask | Welch, the pyramid, several reductions | `Intervals`, one member: the good samples (the complement of the bad segments, `subtract`) |
| a baseline per member as numbers | `extent::standardize` | stays numbers: a measurement over a selection, handed in (not a form) |
| a neighbourhood: a half-width, a margin, a halo (section 7.3) | `timebank::haloOf`, `frame_tiling::sincInterpolate`, `taps`, `surrogate::shapingTaps`, `filterValid`, `wt`'s margin | a `Rule` with hop 1 and width 2h + 1 (origin −h): the planner reads the halo from it; the function keeps its half-width parameter until the planner does |
| a radius or a distance | `ChartKit::disc`, `quadrature::weightedPhaseGradient`, `tracks::link`, `lag::pairsAtDistance` | a `distance` selection, resolved by `resolve_distance` to `Elements` |
| selections made in compute (section 7.5) | `tower::makeFrameTiling`, `frameSpan`, `parent`; `regions::detectRegions`, `cores::corePatches`; `intervals` detection; `lag::pairsAtDistance` | return a `Selection` (a tiling; a set of indices; a set of spans); the pair list is a product's members (section 9) |

## 9. Product selections (2026-11-17)

A measurement per chunk per band, per time tile per cortical tile, or a burst on some channels for some time, lies on a PRODUCT manifold (the schema's
`manifold.type = 'product'`, D37: a recording is on sensors × time). Its selections stay within the five types:

- **`product(factors)`** (`selection.h`) is a `set` whose members are given a fourth way, by FACTORS: each factor a selection on its own manifold, the
  members the tuples of the factors' members. A factor that is not a set is one member, so a single burst on three channels for 200 ms is
  `product(span on time, indices of the three channels)`, one member; a set of such bursts is a `set` of those products. A product of products is
  flattened; a factor with levels is refused (name the level).
- **Identity: the member ordinal is row-major over the factors' ordinals**, the last factor fastest: k = ((i₀·n₁ + i₁)·n₂ + i₂)…, nₓ = `ordinal_count`
  of factor x. This is the layout the computes already use (`AnalyticSums`, `TensorSums`: [tile × member]; `measure.band_amplitude`: chunk × channel ×
  band). `factor_ordinals(p, k)` and `product_ordinal(p, …)` convert.
- **Label: the tuple of the factors' labels, each read with its own factor's lookup** (`factor_labels(p, k)`). A product carries no lookup of its own.
  Why not a lookup of pairs: it would repeat n₀·n₁ entries that the factors already hold, go stale when a factor's lookup is edited, and lose the
  factors' own structure (a band's range, a tile's level); two lookups are the factors' own, unchanged, and a measurement filtered by one factor
  ("every chunk in alpha") is arithmetic on the ordinal, never a string match.
- **Partition**: a product is disjoint when every factor is (`disjoint`), and covers the product manifold when every factor covers its own.
- **Resolved form: `ProductForm`** (`resolve_product(p, spaces)`, one `Space` per factor saying where it lies and which form to take): the factors'
  forms (a `Rule` for the chunks, `Intervals` for the bands' bins, `Codes` for the channels or tiles) and the row-major numbering (`counts`,
  `ordinals(k)`, `ordinal(…)`). A compute loops over the factors' forms and writes value k at the product ordinal; the product is never materialized.
- **Events that are not boxes** (a burst whose channels change over time; a region that moves) are not products: they are `indices` (or a set by codes,
  `Codes`) on the product manifold's cells, numbered row-major as the field's array is (time slow, element fast); `Elements` when they overlap. Section
  6's question 4 is answered this way for compute: a box is a product, any other spatiotemporal shape is a set of cells.

Examples (compute):

```cpp
Selection chunks = tiled({0, 32768 / sfreq, 32768 / sfreq, n_chunks}, "s");            // the chunk partition of time
Selection bands  = set_of_spans({{8, 13, 0}, {13, 30, 1}}, "Hz", {{0, "alpha"}, {1, "beta"}});
Selection cells  = product({chunks, labels(channel_codes), bands});                     // time x channel x band
ProductForm f = resolve_product(cells, {{FormKind::rule, time_line}, {FormKind::codes, std::nullopt, n_channels}, {FormKind::intervals, bin_line}});
```

**How io should store a product** (a description for io's selection CRUD; not built here):
- One `selection` row: `type = 'set'`, `manifold_id` the product manifold (its `geometry_dimension` rows name the factors' manifolds), `cell` one rank
  per factor (`'1,0,1'`: time edges × channel vertices × frequency edges), `n_members` the product of the factors' ordinal counts, `array_path` null (a
  rule: nothing per element is stored), no `dictionary_id` (the factors carry theirs).
- Each factor is a stored selection in its own right, on its own manifold, with its own dictionary. The product names them in order in a new detail
  table, the shape of `operator_factor`: `selection_factor (selection_id, ordinal, factor_selection_id)`, primary key (selection_id, ordinal).
- Measurements: `selection_measurement.code` is the product's member ORDINAL (row-major); an array-backed measurement is `[rows × members]` with the
  members in the same order. "Every chunk in beta" and `within` one factor's member are arithmetic on the ordinal (k / stride mod nₓ), which io's
  measurement query does from the factors' counts; the roll-up `coarser(shift)` applies to one factor's ordinal (unpack, shift, repack).
- A set of products (a set of box events) is a `set` whose members are product rows (`member_of_id`, `member_ordinal`), each with its factor rows; a
  shape that is not a box is `indices` or codes on the product manifold (an array, as today).
