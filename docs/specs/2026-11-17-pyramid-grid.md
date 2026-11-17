# The measurement pyramid as a grid on cortex x time (spec, 2026-11-17)

The pyramid (`run pyramid`) was a hand-wired C++ verb (`pyramid_build`: a hand-made estimate, a hand-made frame tiling, bespoke accumulators over a
tile-per-sample array, a hand-written store). It is now a definition of registered nodes on the selection system
(`docs/specs/2026-11-17-selection-module.md`, sections 8 and 9). This spec is the graph, written first, and its check against the Studio's tower and
Rheome's joint ladder.

## 1. What it is

The pyramid is a GRID on the product manifold cortex x time: dyadic spatial tiles (the **leaves**, one level of a surface's tile ladder) x time tiles
(the **frames**, the cycles of ONE tower level). Its cells are the Studio's D129 joint partition, `nxr::selection::joint({frames, leaves})`: first factor
(the frames) fastest, its radix rounded up to 2^bits, code = `(tile << bits) | frame`. A grid is defined before data, its hierarchy is in closed form
(a tile's parent `code >> 1`; a frame's level-m tile `tower code >> m`) and roll-ups are only along its grid factors, at read.

Every number it stores is a MEASUREMENT over a member of that grid (the cardinal rule: no node reduces on its own):

| measure (per band) | meaning |
|---|---|
| `power_sum` | sum over the cell's samples and sources of area x \|J\|^2 |
| `envelope_sum` | the same with \|J\| |
| `envelope_max` | max \|J\| over the cell |
| `tensor_11 ... tensor_total` | a free-orientation kernel's Hermitian orientation tensor in the canonical frames (nine numbers and the trace), instead of the three above |
| `count` (over the frame grid, not per band) | the good samples of each frame: the weight every mean divides by |

Only mergeable ingredients are stored (sums add, maxima merge): means, densities and every ratio are derived at read. J = K z, z the band's analytic
signal at the sensors (the inverse is linear, so the filter comes first, D106).

**Coupled levels (settled, items 5 and 10).** A band selects (it is not a frequency axis; item 11: one measurement set per band, each on its own cells,
no chunk x band product). The leaves' level and the frame level are CHOSEN here by the caller (`--depth`, `--frame-level`); the level a band would set
on its own is compute's `grid.h` (`time_level_of_band`, the Studio's `fitLevelOf`; `space_level_of_band`, Weyl) and is what a caller reads off to choose
them. The frame level is also derivable from a band by `run frametiling --band` (the coarsest tower level whose frame is at most one cycle / 36).

## 2. The graph

```
before
  joint.cells        (space = leaves, time = frames)               -> joint_space (cortex x time), cells        the grid, closed form (compute: selection::joint)
  pyramid.estimate   (kernel, frames, leaves)                      -> estimate                                  a Field with no bytes, its provenance
chunks over recording (io's chunk service: one chunk and its halos)
  pyramid.frame_samples (recording, frames, bad)                   -> count                                     measurement over the frame grid
  for each band b of the bank
    pyramid.band_analytic (recording, kernel's channels)           -> z_b        pointwise, windowed: block spectrum x band gain, zero phase (halo)
    pyramid.band_sums     (z_b, kernel, leaves, frames, surface,   -> sums_b     accumulate: J = K z_b blocked over samples; measurements over the cells
                           canonical, bad)
outputs   joint_space (product manifold), cells (selection), estimate (field), count (measurement over the frame grid),
          sums_b (measurement over the cells, of the estimate, within the band bank, code = the band's index)
```

Ports are typed (`types.md`): `z_b` is a complex streamed field on (kernel's channels x time); `pyramid.band_sums` takes the frames and the leaves as GRIDS
(`frames` a tiling, `leaves` a dyadic set: `partition = true`) and its output is a measurement over the members of the cells. The band banks are made before
by `run pyramid_bands` (`pyramid.bands`: spans on the frequency line, code = index) and `run spatial_bands` (`pyramid.spatial_bands`: one span of the
eigenvalue line a ladder level, its Weyl band).

**Compute** (`compute/include/nxr/pyramid.h`, `frame_grid.h`): `pyramid::frameOfSamples` and `addSampleCounts` take the frame grid RESOLVED on the line
(`selection::Intervals`, one interval a frame) instead of a `tower::FrameTiling`; `bandSums` and `tensorBandSums` take the leaves as a
`selection::MemberOfElement` (the member number of each source row) and the frame of each sample (the block's member of the time grid) instead of a
`codes` pointer and a `tile` array. `tower::frameGrid` is the grid maker (`selection::tiled`, origin = the tower's `origin_s`, width = hop = a frame);
`tower::frameIntervals` resolves it on the recording's line. A frame is not a whole number of samples (395.5 at 1200 Hz): the grid's rounding rule
(a cell [a, b) holds the points `[ceil(a), ceil(b))`) puts each sample in the frame `floor((offset + i) / (frame fs))`, the tower's and nxr's.

## 3. Checked against the references

| Studio `engine/toolbox/tower/tower.ts` | here |
|---|---|
| SPACE level j: 2^j equal-mass tiles, lambda band [4 pi 2^(j-1)/A, 4 pi 2^j/A] (Weyl), one index for tiles and bands | the leaves are a level of the ladder; the spatial band bank (`pyramid.spatial_bands`) is the same Weyl band per level, code = the level |
| TIME level k: dyadic segments of 2^(K-k) samples | the frames are a tower level (86 400 x 2^L s: dyadic in time as the chunk grid is); level m is `code >> m` |
| SPEED the tower's diagonals, derived never stored | not stored: a speed is derived from a (leaf, frame) pair's size and duration at read |
| cell (j, k) | a joint cell (leaf, frame): `(tile << bits) | frame` |

| Rheome `+geom/jointladder.m` | here |
|---|---|
| "A ROW IS A SELECTION, NOT A MEASUREMENT": a row says what CAN be measured | the cells are a selection (`joint.cells`, closed form); the sums are the measurements over it |
| power rolls up exactly to coarser rows; a trajectory or speed does not | only sums and maxima are stored and rolled up; ratios (speed, gradient, density) are derived at read |
| depth x octave x frame level | depth = the leaves' level, octave = the band's bank entry, frame level = the tower level |
| `jointcell`: a feature's depth from its extent (coarsest depth whose tile holds it; refusal with the reason) | not used here (it addresses irregular members on the grid: `selection::dyadic_address`, grid.h) |

## 4. What changed for the caller

- `nxr-neuro run pyramid` is a definition (`--help` lists its parameters). The leaves (`run tiles`), the frames (`run frametiling`) and the band bank
  (`run pyramid_bands`) come first, as `run joint` takes its levels; `--depth` has no default (it was the finest level every piece reaches).
  `pyramid build` and `pyramid spatial-bands` are gone (`run spatial_bands`).
- The estimate, the joint cells and the measurement rows are stored by the declared outputs (`docs/specs/2026-11-17-definition-outputs.md`).
- Removed: `pyramid_build` and its options (pipelines/src/pyramid.cpp). Kept for the verbs not yet converted: `pyramid_leaves`, `pyramid_leaves_record`,
  `frame_cells`, `frame_sums_write`, `pyramid_bands_selection`, `pyramid_spatial_bands`.

## 5. What it cannot resolve (also in the estimate's `producer.warnings`)

The minimum-norm current is smooth: its resolution is the kernel's point spread, not the leaves'. A band's analytic amplitude means something only for a
band narrow against its centre. The filter is circular over the block read, so the first and last `halo_seconds` of a recording carry its edge effect. A
Morlet band overlaps its neighbours (power_partition's add to broadband). A free kernel's principal axis is partly the kernel's own prior.

## 6. Decisions for the PI and open needs

1. **Storage location of the arrays.** nxr's atlas node `source-time/<recording>__<kernel>_atlas/<tree>/{power,env,envmax}` ([band, leaf, frame], one array for all bands) is not
   written: the declared measurement output stores `<estimate path>/<measure>`, one array [leaf, frame] a band and measure, found through the rows' `array_path`. nxr's reader
   (`io/test/parity/test_pyramid_parity.py`) and the Studio must follow the rows, or the declared outputs gain a path option. Which?
2. **`depth` has no default and the leaves, frames and bank must exist** (they were made on demand). Keep, or let the runner make missing grids first as it makes missing eigenmodes?
3. **The `atlas` block on the leaves' record** (`areas`, `node`, `frame`) was written into the leaf partition; it is now in the joint cells' `params.atlas` (copied from the leaves'
   record); the roll-up reads it there. The gauge (canonical frames' id) is not recorded.
4. **The Frequency and Eigenvalue lines** live in the dataset's default subject; the declared line outputs make them in the subject the bank is made in. Keep?
5. **One FFT a band** (the bands are repeated by the definition): the spectrum is taken once a band, not once for all bands (about 2x the transform cost; the result is unchanged).
6. The bank's dictionary attributes (`low_hz`, `high_hz`, centre) are not written (the spans carry the edges): the declared selection has names only.
7. `pyramid reduce` (group.cpp) and the leaf/frame sums of `phasegradient` and `opticalflow` stay hand-wired until their agents convert them.

## 7. Needs outside this agent's files

- io: `FieldOut::by_operator_id`, `MeasurementOut::within_selection_id/within_code` (added, small); declared outputs keys `within`, `by_operator`, field `derived_from` (added).
- Tests to port in the final round: `compute/test/test_pyramid.cpp`, `test_pyramid_threads.cpp` (new signatures), `pipelines/test/test_pyramid_pipeline.cpp`,
  `test_pyramid_reduce.cpp` (uses `pyramid_build`), `test_spatial_bands.cpp`, and new: `frameGrid`/`frameIntervals` against `tower::frameOfSample`,
  `pyramid.band_sums` against `bandSums`, the definition's dry run.
