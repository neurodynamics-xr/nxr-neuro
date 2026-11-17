# The Selection module (spec, 2026-11-17: in discussion)

Status: the vocabulary and grammar below are agreed in direction; the open questions at the end are being decided. Nothing here is built yet beyond the io
selection entity (`io/include/nxr/io/selection.h`, a port of the Studio's `entities/selection.ts`).

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
field's shape without compute. io's constant-Q writer still designs its bands; the design moves to compute in the CRUD step.

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
