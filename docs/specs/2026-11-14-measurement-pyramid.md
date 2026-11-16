# The measurement pyramid: every reduction a measurement on the dyadic tiles (spec, 2026-11-14)

**Name:** the system is the *measurement pyramid* in nxr-neuro (code, commands, options, docs). "Atlas" is an artifact of
Rheome's "atlas" and is not used for new nxr-neuro names. The STORED nxr format keeps nxr's keys and names unchanged for compatibility
(`params_json.atlas.*`, `<partition>_atlas` nodes, the `atlas bands` Selection, the frame tiling's params): nxr-neuro writes exactly what nxr
writes there. Below, "nxr's atlas" names the reference design being ported.

## Why

Principle (2026-11-14): every reduction is a measurement on the dyadic spatiotemporal tile system, a pyramid in space (vertex to the whole cortex), time
(frame to the whole run, or time only with no spatial extent, as burst detections), frequency band and spatial scale; the dyadic tiling places each
measurement by its tile code.

nxr-neuro has the pieces (the surface ladder, constant-Q levels, joint Selections, array-backed measurements, a lookup that rolls scalar rows up by
`code >> shift`) but not the measurement pyramid as one system. Meanwhile its summary-map commands store reductions as Fields, one per variant: on the fixture
2 031 `phasegradient` and 420 `cyclepattern` Fields (finished ratios and eigenvectors, no ingredients), which is also most of the ~7 400 files of one
subject. And the frame tiling added for vortex cores (36 frames a cycle of the band's centre) breaks the dyadic time nesting (my error: the time-tiles
rule was not checked).

This spec ports nxr's atlas (the nxr format's design for this) as nxr-neuro's measurement pyramid, and brings every reduction into it. It is the same format: what nxr-neuro writes, nxr
and the Studio read.

## The design ported (nxr; decisions D127-D135, D141, D157, D158, D163 in the Studio's decision register)

- **There is no atlas format** (D129, D134, D163): an atlas is Selections plus array-backed measurements on them.
- **Space:** a tree partition (`set` Selection `<surface> tiles L<D>`, or `group tiles` on the template through the registered sphere, D128), leaf code
  `h·2^D + t`; a node at level ℓ is `code >> (D − ℓ)`; ℓ = −1 is the whole cortex. `w` = leaf areas; `lc/L<k>` the Levi-Civita connection per chosen
  level (explicit, never rolled up).
- **Time: the tower** (D135): level L is a cycle of 86 400·2^L s (the day is level 0); frames are tower cycles at `frame_level` (default −18,
  ≈ 0.33 s), frame k = code `frame_code0 + k`, a level-m tile = `code >> m`, anchored on the recording (or day / birth). Frames are absolute time, not
  per band. Per-frame good-sample `count` on the frame tiling; a bad sample carries nothing, so every sum stays mergeable.
- **Frequency:** the `atlas bands` `spans` Selection on the default subject's Frequency manifold (D158); a measurement's band is `within_selection_id`
  + `within_code`. Default bank: octaves 1-64 Hz, a power partition (band powers add to broadband).
- **Joint:** one `set` Selection per recording × kernel × tree on cortex × time, code `(tile << B) | frame`, `layout: [band, tile, frame]`.
- **Stored: only mergeable ingredients per leaf** (sum, max, min: `w, s1, s2, n, min, max`, `power, env, envmax`, the free-orientation tensor in
  canonical frames, gradient sums), arrays [band, leaf, frame] in the partition's node (`atlas_node`, D157).
- **Roll-up at read** (D132; nxr `measure/rollup.py`, the Studio's `db/rollup.ts`): space by reshape, time aligned on the tower (`code0 % 2^m` neutral
  front), only sum/max/min; means, variances, densities derived from rolled-up sums. Path-dependent quantities (transport, trajectories) are explicit
  per level, never rolled up.
- **Group** (D130, D131): `reduce` sums subjects cell for cell on the group tree into the default subject's partition node; group time collapses frames.
- A job runner already runs it, per subject and for the group: the map-then-reduce of the dataset-wide spec.

## Where nxr-neuro goes beyond nxr (decided: the maintainers approved every recommendation below, 2026-11-14)

1. **The vertex level.** nxr's leaves are tiles at depth D (default 8: 512 leaves); there is no per-vertex storage. The the maintainers' pyramid starts at the
   vertex. Options:
   - (a) leaves at the finest ladder level (each tile a few vertices; the ladder stops where every tile still holds a vertex), same format;
   - (b) a vertex leaf level (each vertex its own member, codes extended below the finest tile), a format extension;
   - recommended: (a) as the default, the leaf depth a parameter; (b) only if a measure needs single-vertex resolution, decided then.
2. **The spatial-scale (eigenmode, Weyl) axis.** Not in nxr's atlas. Ported as D158 ports frequency: a `spatial bands` `spans` Selection on the
   eigenvalue axis (λ, 1/m²), one span per Weyl level, a measurement's level as `within` code. Recommended: yes; the arrays gain a leading axis
   [band, spatial band, leaf, frame] only where a measure has one.
3. **Frames on the tower for phase-resolved events.** Vortex cores and spatial events need frames far finer than 0.33 s (the project's 10° steps: 2.3 ms
   at 12 Hz). Recommended: the frame tiling becomes tower frames at a chosen level (the finest tower level at or under the band's period / frames per
   cycle: level −25 ≈ 2.57 ms for alpha), so every frame has a dyadic parent; `frametiling compute` takes `--frame-level` (or derives it from the band
   and `--frames-per-cycle`, snapped), and the cores and spatial-events steps run on it. This replaces PR #144's per-band tiling.
4. **The band filter.** nxr filters per band by one FFT over the whole record; nxr-neuro streams one chunk at a time (method rule 6). The port uses
   nxr-neuro's chunked analytic step (halo, Morlet default) with a stated tolerance against nxr's whole-record filter in the parity test, and nxr's
   power-partition bank as the named option `shape: power_partition` so the band powers still add to broadband when that is wanted.
5. **Variants in the key.** The measurement key has no method or level beyond `within`. Band and spatial band go in `within` (above); a method
   variant is a different measure name or recipe (`computed_by`), as nxr does. Recommended: no schema change.

## Bringing the summary-map commands in

Each stores its additive ingredients as measurement-pyramid measurements per leaf (and band, spatial band, frame where it has them); ratios move to read:

| command | stores today (Fields) | stores after (measurements) | derived at read |
|---|---|---|---|
| `phasegradient` | H/P, velocity, speed, travelling index, P, Q; with a spatial band: timing, propagation, stream function, divergence, curl, inconsistency, mode coefficients | leaf sums of the phase current H (in canonical frames), P, Q, and the per-leaf form ingredients | gradient H/P, speed, travelling index, curl and divergence densities |
| `cyclepattern` | patterns × {stream, potential, normal}, coefficients | nothing new: its input, the windowed cross-spectrum sum, is already a statistic | the patterns (a decomposition of a statistic), at read, or one statistic node per decomposition if kept |
| `opticalflow` | time-mean divergence, curl, speed, velocity | leaf × frame sums | the means and speeds |
| `band_power_source_map` | a mean-square Field | the pyramid's `power` (its own measure) | mean power, density |
| `psd compute` | `<recording>_psd_welch` Field | a statistic (as `sourcepsd`); already so since 2026-11-13 (checked 2026-11-14) | |
| `tile_summaries_write` | `summary/<stem>/…` arrays outside the table | the same sums as measurement rows on the partition | |

Fields remain what they are for: recordings, operators, per-frame materialised series (e.g. `flow_modes`), and the source estimate as a byte-less
Field (stored as its reductions only).

## Also in this change

- **The lookup rolls arrays up:** `query_measurements` gains the pyramid's read-time roll-up for array-backed measurements (space, time on the tower,
  band and spatial band slices), porting `rollSpace` / `rollTime` / `derive`.
- **Development runs stay out of the fixture:** they go to a development copy or a staging dataset. The 2 445 development Fields already in the fixture
  stay until the maintainers decide (deleting data).
- **Standard zarr, fewer files:** the chunk key separator `.` (c.0.0 in the array's folder: −5 287 folders on the fixture subject), large arrays
  sharded (~64 MB), every reader reading either separator. The small-file store (dataset-wide-runs spec, section 1) is deferred until measured on a
  subject without development clutter.

## Order (each a PR, parity with nxr on synthetic data)

1. The tower and tower frames (`frametiling` on the tower; cores and spatial events moved onto it).
2. The pyramid core: tree partition, `atlas bands`, joint Selection, `pyramid build` (power, env, envmax; then the free-orientation tensor), parity with
   nxr's `atlas build` on a synthetic recording and sphere.
3. Read-time roll-up for array-backed measurements in the lookup.
4. The spatial-bands axis (decision 2).
5. `phasegradient`, `opticalflow`, `band_power_source_map`, `psd`, tile summaries moved into the pyramid; `cyclepattern` reads its patterns from the
   statistic.
6. The group reduce (the dataset-wide spec's group scope is this).
7. The zarr separator and sharding (a format version; every implementation reads both).

## Checklist (the architecture policy)

- Compute: the band filter, the leaf operator, the tensor and the roll-up are pure functions of arrays.
- io stores Selections and measurements; reductions are measurements over partitions; ratios are derived at read.
- The caller names the recording, kernel, tree depth, bands and spatial bands.

## Decisions after steps 2-5 (the maintainers delegated them, 2026-11-14)

1. **`phasegradient --spatial-band`'s non-additive quantities:** divergence and curl become exact tile flux and circulation (nxr `measure/integrals.py`),
   pyramid measurements; timing, stream function, propagation, inconsistency and mode coefficients (global fits over the modes, or functions of the
   phase of the edge lags: no additive form) become ONE statistic node per frequency band × spatial band, with vertex and quantity axes, of the
   cross-spectrum statistic they are computed from. No per-variant Fields.
2. **Leaf vectors are summed in each vertex's canonical frame** (nxr's convention, `grad/{c1,c2}`), stated in the warnings (a leaf larger than the
   wavelength averages a turning wave out). The Levi-Civita-transported sum (to the leaf centre) is a later named option, with the `lc` levels.
3. **`band_power_source_map` is retired** in favour of `pyramid build` (leaf resolution; analytic-signal power, a factor 2 above a real band-pass's
   mean square: documented). The definition and its runner tests move to a pyramid definition.
4. **Maps of a statistic's slice** (`quadrature --tiles`): each gets a byte-less Field (the pattern of the source estimate and `phasegradient`), so
   its rows satisfy the schema's array-backed rule. No schema change.
5. **`cyclepattern`'s mode coefficients and lags are statistics** (confirmed).
6. **The singular-vertex `total`:** a vertex with no canonical frame is out of the nine tensor sums, and its whole |J|² (normal share included) is in
   `total`. Fixed in nxr-neuro and in nxr together (a correction of nxr's convention, which dropped the normal share), with the parity golden
   regenerated.
