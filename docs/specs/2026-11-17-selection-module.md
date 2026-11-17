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

THE MODULE (decided 2026-11-17): selections and partitions of manifolds are their own library, `selection/` (`nxr_selection`, namespace `nxr::selection`),
below io and compute, which both call on it; it depends on neither. Order: time partitions first, then the surface (dyadic tiles, Weyl), then the joint case (the product).

THE TIME PARTITION (`selection/include/nxr/selection/time_partition.h`): `time_partition(samples, sfreq, ConstantQ)` gives a recording's levels, each the
partition of its samples into tiles of 2^k samples from the first sample (a tile by arithmetic, j = sample >> k), named by k, and the band each level
resolves; constant-Q chooses which k exist (the levels its bands need). The chunk grid and a constant-Q level are the same partition at their k; other
methods come later as options of the same function. io's constant-Q writer takes its bands from it.

Layers around it: io (CRUD and resolution); compute (pure makers and reducers: tiling arithmetic, the dyadic ladder, product codes, geodesic discs and kernels by
heat distance, log-map boxes, set combinators, reductions); pipelines (`select.<type>` makers, `measure.over`, the YAML keys `at`, `over`, `within`); the
command line and the bindings with the same verbs.

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
