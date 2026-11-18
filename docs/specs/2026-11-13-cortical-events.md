# Events on the cortex: patches on a static map first, then space × time (spec, 2026-11-13; PLAN, a DRAFT for the maintainers)

## Why

Plan (2026-11-13): the equivalent of burst detection on the cortical surface, using the same idea with Selections, graph wavelets,
measurements/reductions and scalar summaries, and build it first on a STATIC map (e.g. a PET map, no time), then tie it to time for space × time events.
The time design it mirrors: `docs/specs/2026-11-13-time-events.md`, `2026-11-13-burst-statistics.md`. The references: the audit
`docs/audits/2026-11-13-atlas-joint-cells.md` (the Studio's atlas, Rheome's joint ladder).

## The rules

- **A spatial level IS its spatial band** (the D84 analogue): level l of the surface ladder (2^l equal-area tiles per hemisphere) is paired with its
  Weyl band (λ = 4π·2^l / A). A map is filtered to the level's band before it is measured on the level's tiles: a level holds the patterns of its own
  size, as a constant-Q time level holds its own band. Each level is therefore computed and stored on its own (a coarser level is NOT a roll-up of a
  finer one); within a level, roll-ups are exact as ever.
- **The same detection rule as in time.** An event is a CONNECTED set of tiles above the low threshold that somewhere reaches the high threshold
  (hysteresis; with enter = exit, plain connected regions above one level), merged across small gaps, kept within size limits.
- **Only the neighbourhood differs.** In time, a tile's neighbours are the previous and the next (a line graph, implicit in the order). On the cortex,
  two tiles of a level are neighbours when a mesh edge joins them: the graph already exists, `nxr::tileAdjacency` (`compute/include/nxr/ladder.h`, nxr's
  `tile_adjacency`; it is the mesh's vertex graph, the Laplacian's pattern, contracted by tile). Hemispheres never touch.
- **Channels are another manifold** (sensor space): in `band_bursts` each channel is an independent series. They have no counterpart on the cortex.
- **Tools, not instrument limits:** no resolution limit, warning or refusal is built into a block or a generic pipeline; such a limit is a named option
  of a specific detection pipeline, when the maintainers ask for it.

## Phase 1: patches on a static map (first)

The graph (pure nodes; preparation binds the mesh operators, the ladder level and its tile graph once):

```
map (a scalar Field on the surface, e.g. PET)
  → spatial_band   the level's Weyl band (graph wavelet: nxr::filterbank weyl; stored eigenmodes first, Chebyshev an announced fallback)       ℝ per vertex → ℝ per vertex
  → measure        area-weighted mean over the level's tiles (nxr::partition::accumulate)              → ℝ per tile     [stored: measurement]
  → summarize      the baseline across the level's tiles: median, mad                                  → statistic      [stored]
  → detect_regions hysteresis on the tile graph (tileAdjacency of the level), merge, keep              → a labelling    [stored: Selection]
  → rollup         each patch's measures from its tiles; geometry (area, centre)                       → measurement over the patches [stored]
  → summarize      statistics across the patches                                                       → statistic      [stored]
```

1. **The band.** The map filtered to level l's Weyl band (a band-pass at the level's scale: features about a tile's size stand out, broad gradients
   and vertex-scale speckle do not; Rheome's `blobscale` is the same idea on a Mexican-hat ladder). The bank and its options are the existing
   `nxr::filterbank` ones (the caller's).
2. **The measurement.** Per level-l tile, the area-weighted mean of the filtered map (`mean`, the registry's), stored over the level's tiling
   (`selection_measurement`, as `measure compute` stores parcel means). Also `max` and `min` per tile, for the roll-up of a patch's peak.
3. **The baseline.** One map has no time axis, so the baseline is across the level's tiles: median and robust SD (1.4826 MAD), a statistic
   (`summarize`), stored and read by the detection, as `band_bursts` reads its baseline.
4. **The sign** (new: a band's amplitude in time is never negative; a band-passed map is). A named option `sign`: `positive` (bumps: e.g. high uptake),
   `negative` (dips), `magnitude` (|x|). The caller chooses.
5. **Detection, `detect_regions`.** One compute function on a GRAPH: values per node, a list of neighbour pairs, the thresholds → a label per node
   (−1 for none). Tiles ≥ median + exit_sd·SD form connected regions over the pairs; a region is kept when it holds a tile ≥ median + enter_sd·SD and
   min_area ≤ its area ≤ max_area; regions within merge_gap tiles of each other joined (or no merge, the default for space: DECISION 2). Time's
   `detect_spans` is this function on the line graph (k, k+1); folding it in is a later refactor, not this phase.
6. **The output: a sparse Selection on the surface**, a labelling of the level's tiles (a `set` Selection on the surface: a code per vertex = its
   patch's ordinal through its tile, −1 for none, the model's rule that a negative code is in no member). How it was made in its `params_json`
   (pipeline, level, band, thresholds, the baseline statistic, cache key), as the bursts' Selection records it.
7. **Each patch measured** by the roll-up of its tiles (`nxr::partition::accumulate` over the labelling, or the same roll-up as `rollup_events`):
   area (Σ tile areas), mean (area-weighted), peak (max of the tile maxima), centre (area-weighted, the vertex nearest it), number of tiles; which
   parcels it overlaps through the lookup.
8. **Statistics across the patches** (`summarize`): count, the distributions of area, mean and peak.

**Tests (synthetic only):** a sphere with planted bumps of known size and height on a smooth background: at the level whose band matches the bumps'
size, each bump found as one patch of the right tiles, with no others; area and peak against direct; a bump at another size not found at this level;
`sign` negative finds planted dips; a rerun is a hit.

**What it gives:** where a map stands out from its surroundings at the scale of level l (where, how big, how strong), and over several levels, at
several sizes.

## Phase 2: space × time (after phase 1)

- The same chain on a source estimate's band amplitude (time band → kernel → spatial band → |z|), measured per (spatial tile × time tile): the
  windowed features of the recording on the cortex.
- The graph gains the time links: a tile is also joined to itself in the previous and next time tile. `detect_regions` runs unchanged on that graph.
- The baseline per spatial tile over the time tiles (each tile's own history, as in time), or across tiles (DECISION 3).
- An event is then a patch over a stretch of time: where, how big, when, how long, how its area and centre change.
- The joint code ports the Studio's atlas layout (code = (tile << B) | time tile), per (spatial level, time level) pair (the audit).

## Global bursts: WHEN, from the cortex

"We don't work with per-channel bursts. Bursts should be global. ... the time-domain burst detection is there to tell us approximately when something
happened and the space for where something happened." A mean over channels is refused (it washes out local increases). Adopted: Rheome's rule
(`rheome.flow.labelalpha`, EPISODE family, `+flow/labelalpha.m:66-74, 118-132`), audited 2026-11-13:
- **The trace:** per time tile, the MAX over the spatial tiles of one level of the tile amplitudes (a reduction over the spatial partition, max): the
  strongest tile anywhere, so a local increase stays whole. Rheome: tile means of the source alpha envelope at depth 7 (~33 mm), clipped at 0, a
  0.1 s moving mean (here: the time tile itself), max over tiles.
- **Whole brain by default** (the max over both hemispheres' tiles); **per hemisphere** a named option (Rheome's form: two event sets).
- **The tiles' signal: the spatial level IS its spatial band** : the tile amplitudes are of the field filtered to the level's Weyl band, as in
  phase 1 and phase 2 (Rheome uses the plain tile mean; this departs from it by the maintainers' rule).
- **Detection:** `detect_spans` on the trace, unchanged (Rheome's 1.5 / 1.0 robust SD, merge 0.1 s, ≥ 0.4 s).
- **Edges:** Rheome's rise and fall fit (`detect.risefall`: tOn, tOff, τR, τF, plateau, r²; kept at r² ≥ 0.7 in labelalpha) PORTED, as per-burst
  measurements.
- **Order:** phase 1 (static patches) → phase 2 (tile features over space × time) → global bursts from the max over tiles → per-burst cross-spectrum.
  `band_bursts` (sensors) stays as it is; it is not the global detector.

## Phase 3: from the cross-spectrum to patterns common to bursts (outline; a spec of its own before building)

The lessons of the lag work (vertex-scale edge lags tiny and noisy; windows not matched to events; phases averaged; medians of bimodal distributions)
answered by the design:
1. **Each global burst's cross-spectrum** = the exact sum of its time tiles' stored cross-spectra (`crossspectrum compute --estimator analytic`, Σ z zᴴ
   per tile at the band's level): the burst is the window, no new pass.
2. **Each burst's cycle pattern at a declared spatial level J** (`cyclepattern`, `--spatial-band weyl:J`): leading pattern, power share, standing share.
3. **Patterns common to bursts:** the leading patterns of the bursts' summed cross-spectrum; each burst's similarity to them (a measurement per burst);
   their distributions across bursts (statistics).
4. **Per level-J tile per burst:** phase current and power sums (products only, never phases or complex values summed); direction, speed, travelling
   index derived at read; across bursts per tile the consistency |Σ_b J_b| / Σ_b |J_b| and the distributions.
No nulls, tests or averaging unless the maintainers ask; distributions, never a lone median.

## DECISIONS for the maintainers

1. **DECIDED (2026-11-13; the maintainers): the Weyl pairing.** A level-l tile is about one bump of its band across
   (half a wavelength ≈ 0.9·√(A/2^l); a tile ≈ 1.1·√(A/2^l)), so the tile mean of the band-passed map IS the bump's height: the matched blob detector.
   Rheome's ≥ 2 wavelengths per tile serves a different measure, a band's ENERGY in a patch (a signed mean over several wavelengths cancels to ~0); band
   energy (mean x² over a coarser level's tiles) is a named option of a specific pipeline, not phase 1.
2. **DECIDED: no merging across gaps by default** (patches that do not touch are separate); merging is a named option.
3. **The space × time baseline:** per spatial tile over time, or across tiles.

## Phase 1 build notes: issues for phase 2

Built as `patches compute` (`pipelines/src/patches.cpp`), `nxr::regions::detectRegions` (`compute/include/nxr/regions.h`), `filterbank::analysisMember`, tests `test_regions` and
`test_patches_pipeline`. What the build met that the space × time phase must account for (each: what happened, why it matters, options):

**The runner and the shape of the pipeline**
- **A static map does not fit the runner's definitions.** The runner stores `measurement`, `spans` and `statistic` outputs only of the recording the chunk loop reads (`plan`: "this definition has
  no chunk loop"), and its steps `measure`, `detect_spans`, `rollup_events` are time-partition steps. So phase 1 is a verb that composes the existing verbs (`measure_compute`, `statistic_compute`)
  and the pure nodes, with no generic executor. Phase 2 has a chunk loop (the source series), so a definition is possible, but needs: a `measure` over a spatial or joint partition, a
  `detect_regions` step taking the graph, a `set` Selection output kind (a labelling), and a roll-up over regions. Options: extend the runner with those (the architecture's route; a larger change), or
  keep a verb that wraps the chain. Decide before phase 2 is written; phase 1's nodes are already the pure functions either would call.
- **No generic graph executor for the verb.** The nodes are called in order by hand (preparation → nodes → readback), not by an executor that type-checks edges. The checks are in preparation (scalar Field,
  surface manifold, ladder and level exist, the band's λ range equals the level's, no holes), but "each node's output against the next node's input" is by construction here, not enforced. Phase 2 has more
  nodes and two partitions; an executor earns its place there.

**Band-passed vs plain tile means (the keying question)**
- **What happened.** `measure compute` keys a row by (selection, member, of_field, measure) with no recipe in the key; a band-passed tile mean of the map over `<surface> tiles L<l>` would collide with
  (or silently replace) the plain tile mean of the same Field. **Chosen:** the band-passed map is MATERIALIZED as its own Field `<map>_band_L<l>` and every row is OF THAT FIELD (the architecture's
  materialize-or-reduce rule: the node's output is a Field; the reduction is over it). Nothing of the map's own rows is touched (tested).
- **Why it matters for phase 2.** On the time side the level IS the band and the row is of the recording with the measure `amplitude`, the band implied by the level (D84). For a source series
  over time a materialized band-passed Field per level is a [vertices × samples] array per level: not affordable. Phase 2 reduces the band's pointwise outputs per block (never stores them), so the row
  would be of the SOURCE Field and the (spatial level, time level) band must be carried by the row's key. Options: (a) a band qualifier on the row, as `within_selection_id`/`within_code` carries a
  channel group (the band = a member of a band Selection, as the Studio's atlas does for TEMPORAL bands, audit 1.5; it has no spatial band); (b) one measure name per band (`band_mean`, … in
  `measures_nxr.json`), rejected as it multiplies names; (c) a stored band Field only for static maps, as now. Needs the maintainers' choice; (a) is the Studio's own form.
- **Storage per level.** The band Field is float64 [V] per level per map: 160 kB on a 20k-vertex cortex; fine for maps, the reason it cannot be the form for time series.

**The spatial band**
- **The pairing holds by construction and is checked each run:** level l's λ range in the ladder record (`lambda_min`, `lambda_max`, from the mean piece area) is member l of `weyl(mean piece area, l + 1)`
  (scaling for l = 0). A ladder made with another area is refused with the two ranges. The group tree's ladder carries the default subject's area: a subject's own map is read against ITS surface's ladder.
- **Coarse levels are global.** At level 3 (8 tiles a hemisphere) the band holds a handful of modes, and a bump answers at the antipode (met while building the test). A level-l patch is local only when
  the band has many modes: about level 5 and finer on a hemisphere-sized surface. Phase 2 should say which levels it supports, or refuse coarse levels for events.
- **The polynomial transform is costly and level-dependent.** The Chebyshev degree needed to follow a Meyer band to 1% grows as the band narrows against λmax: 640 at level 5 and 320 at level 4 on a
  5124-vertex mesh, more than 1280 at level 1. The degree is chosen from a stated tolerance (a refusal beyond 5120), so a real cortex (λmax larger) may refuse its coarse levels. The `exact` transform
  is a dense solve, capped at 4096 vertices (unusable on a cortex). For phase 2 the right transform is the stored Hodge–Dirac package: project the series on K modes once, every level a gain on the
  coefficients (diagonal), with no per-level or per-sample polynomial. That also gives all levels in one pass; phase 1 recomputes the band per level (the rule: a level is its own band, no roll-up).
- **The tile mean of a band is not shift-invariant.** A bump on a tile boundary splits over two tiles (the test's dip patch has one or two tiles depending on the texture), and its tile mean is lower than a
  centred bump's. The matched-detector reading of decision 1 holds on average, not per bump. Options: detect on the band at vertices (smoothed by the tile graph), a half-tile-shifted second tiling, or accept.
- **Side lobes.** A band-passed bump has an opposite-sign ring of about half its height, so a strong dip makes a positive ring patch at the default 1.5 SD (the test uses 3 SD). Options for the maintainers: a higher
  default; a rule that drops a patch next to a stronger opposite-sign one; thresholds on the band's energy (x² per coarse tile, the Rheome measure of decision 1's note) instead of its signed mean.

**The baseline and the sign**
- **The baseline is across the level's tiles** (a statistic: median, robust SD). With few tiles (a coarse level: 16 at level 3) it is a rough estimate, and a map that is flat but for a feature gives a
  tiny SD, so thresholds are relative to nothing (a map with no variation beyond round-off is refused; a nearly flat one is not). Phase 2 has a history per tile (DECISION 3).
- **Magnitude's baseline is not the stored statistic.** The stored statistic is of the signed tile means; for `magnitude` the median and SD are of |tile mean| and are computed in preparation with the same
  functions (recorded). The mean of |x| over a tile (`mean_abs`) is a different quantity from |mean of x|. In phase 2 the sign should be a node BEFORE the measure so the summarized quantity is the thresholded one.

**The Selection form**
- **A patch labelling is spatial only.** A `set` Selection with a code per vertex holds a patch that does not move or change. A space × time event changes its extent and centre, so it needs a joint form.
  The Studio's atlas joint partition is a closed-form label (code = (tile << B) | frame) with the data in arrays [band × tile × frame] (audit 1.2); an event's cells could be an array of event ordinals over
  [tile × time tile], or selection rows per (event, time tile). The patches' dictionary holds each patch's tiles as JSON, which will not scale to thousands of cells: cell lists belong in an array.
- **`n_members` counted −1 as a member (fixed).** The writer counted the distinct codes of a `set`, and −1 (in no member) was one. Fixed in the writer (`distinct_count`, `io/src/subject_writer.cpp`: a negative code is no member, the partition arithmetic's rule), tested on a labelling holding −1 (`test_measure_pipeline`); patches no longer repair it. With no patch at all the Selection is still stored (all −1), so a rerun is a hit.
- **Patch ordinals are not stable.** A patch is numbered by its lowest tile, so another threshold renumbers them, and nothing links a patch at level l to the one at level l + 1 (a coarser level is not a roll-up
  of a finer one). Cross-level or cross-time identity needs an explicit link (vertex overlap through the lookup, or tracking: family C), not the ordinal.
- **`statistic` is OF a field.** The model refuses a statistic with no parent, so the patches' area and tile count (the members' own rows, of no Field) are summarized as statistics OF the band Field with
  a flag `--members-own` (`statistic compute`) saying the measure is the members' own. Options: allow a statistic of a partition alone (a model change), or keep the flag.
- **New measures:** `n_tiles` (1012, sum) and `centre_vertex` (1013, a place; none). A joint event's centre is a trajectory (path dependent, not a roll-up), like the Studio's explicit per-level arrays.

**Space × time scale of the work**
- **The tile graph is cheap** (adjacency of a level is read from `lc/L<l>/adjacency`: a few thousand pairs at level 10). The joint graph is (pairs × time tiles) + (tiles × (time tiles − 1)) edges; for an hours-long
  recording `detect_regions` (a global union-find over every pair) must run per chunk with labels carried across the chunk edge (the cardinal rule: no whole-recording accumulator), so its API will want a
  streaming form (a union-find that accepts the next block's pairs, or regions merged at the seam). Phase 1's function is whole-graph.
- **Weights are barycentric areas, fixed** (the ladder's own mass), not an option: an alternative weight (Voronoi) differs by tens of percent per region. If phase 2 offers it, the ladder's stored `w` and the
  measure's weight must change together.

## Phase 2a: the graph (tile features over space × time)

Scope: the windowed FEATURES of a source estimate per CELL = (a spatial tile of level l) × (a time tile of the band's constant-Q level): the band's amplitude and its largest value. No events yet (they
are detection on these cells: phase 2b). Settled by the maintainers; this section records them and the graph, checked against the Studio's chain and the atlas audit before any code.

**The decisions**
1. **The joint partition is the Studio atlas's code layout, ported**: code = (spatial tile << B) | time tile, B = bitsFor(F) = max(1, ceil(log2(max(2, F)))), F the number of time tiles (audit 1.2). ONE joint `set`
   Selection per pair (spatial level l of the surface ladder, time level m of the recording's constant-Q ladder), CLOSED FORM (no per-cell array: `n_members` = tiles × frames, the layout in `params_json.joint`),
   on the product (surface × the recording's time line), made by `joint compute`. The pair names BOTH bands (the level IS the band, D84, in space and in time).
2. **Rows are of the RECORDING with the measure `amplitude` (and `envelope_max`), exactly as the time tiles' rows are** (D96 form: an array beside the recording [rows × members × 2] with the samples counted as
   the last component, two `selection_measurement` rows). Here the Selection is the joint one, the rows of the array are the SPATIAL tiles and its members the TIME tiles: [S × T × 2]. No band qualifier in the key,
   and no materialized band-passed Field (the phase 1 issue about the band-passed row's key is resolved by the pair being the Selection).
3. **Linear maps before any non-linear step (D106).** The spatial band filter F_l is linear and commutes with the temporal filtering, so it is folded into the kernel ONCE in preparation:
   M_l = (F_l ⊗ I) K (F_l on each component of K's rows). No graph filtering per sample.
4. **|z| per vertex is the norm over the components (a free kernel: no orientation assumed)**; the cell's amplitude is the AREA-WEIGHTED mean of |z| over the tile's vertices and the time tile's samples; `envelope_max` the largest |z|.
   A time tile not whole and good is NaN (the time tile rule).
5. **Runner work, kept generic**: the `measure` step takes a joint Selection as `over`; `apply_kernel` takes an optional spatial level; a `magnitude` step. The executor itself gains nothing analysis-specific.

**The graph** (per chunk of the recording, with the halo the band's taps need; steps are the runner's, the nodes pure functions of matrices)

```
recording (ℝ, channels × time)
  → analytic     the time band's analytic signal (the Studio's THE ANALYTIC SIGNAL: even and odd taps of the Meyer band)         ℝ → ℂ, channels × time
  → apply_kernel M_l z  (M_l = (F_l ⊗ I) K, bound once in preparation; F_l = level l's Weyl band)                               ℂ channels → ℂ (vertex × components) × time
  → magnitude    |z| per vertex, the norm over the components                                                                  → ℝ vertex × time          (non-linear: nothing linear may follow, D106)
  → measure      over the joint Selection: per cell, Σ a_v |z| and the max, folded chunk by chunk (accumulators per cell)       → [S × T] amplitude · envelope_max   (stored, array-backed)
```

- **Types checked in the plan, before any data** (the runner's way): the kernel maps FROM the recording's channels TO a surface; the tiles ladder is that surface's; the level exists; the joint Selection names the same
  (space level, time level) and its time level's band IS the analytic step's band (D84) and its tiling is contiguous from the recording's first sample; `magnitude` is the last step before `measure`.
- **Checked against the Studio's chain** (`engine/toolbox/signal/chain.ts`, `studio/engine/gpu/deviceChain.ts`): members → filter (the Meyer band, zero phase) → transform; "a linear map reads the chain BEFORE any
  non-linear transform (D106)": here the band's analytic signal (even and odd taps on the same raw rows) and the kernel with F_l folded in are both linear and come first; |z| is the first non-linear step. The device
  chain's "the analytic signal a scope reads is the chunk FFT's" (here: taps) and its multirate stride (`chainStride`) are not built: the analytic signal is at the recording's rate (see issues below).
- **Checked against the atlas audit**: the code layout and closed form (1.2), the tile level at one depth (1.4: each level is its own run here, the rule), a row's band is carried by the Selection pair (the atlas carries a TEMPORAL
  band by `within`; the atlas has no spatial band, 1.5), the sums folded and merged by max/min/sum (1.6: `amplitude` is stored as the mean, with the area (the level's `w`) and the samples (`count`) the roll-up weights).
- **Reduction rule** (cardinal): the chain is pointwise; the reduction is the measurement over the joint partition (a time partition × a space partition), never a bespoke loop in a node: the accumulator step is the registry's
  `measure`, as for time tiles.

**What is held** (stated in the options' comments): M_l [vertices × components × channels] float64 (a 20k-vertex free cortex and 300 channels: 144 MB); the chunk's block [channels × (chunk + halo)] and its map [vertices ×
components × (chunk + halo)] float32; the accumulators 2 doubles per cell and the samples a time tile: S × T × 16 bytes (level 6 and 600 time tiles: 1.2 MB). The recording is read by the chunk service only.

## Phase 2a build notes: issues for phase 2b

Built: `joint compute` (`pipelines/src/joint.cpp`), `run cortical_cells`, `apply_kernel` with `tiles`/`level`, `magnitude`, `measure` over a joint Selection, `spatial_band.h` (shared with patches), test `test_cortical_cells`.
The runner took the work without a change to the executor (the three registry steps' plan/prepare/fold/finish and the generic `store_measurement`, whose array is [rows × members × 2]: here rows = spatial tiles, members = time tiles).

- **The joint Selection is made by a verb, found by the definition.** The plan cannot create a Selection (it is pure), and `store_measurement` needs the Selection id before the run (the hit check), so `joint compute` writes it and the
  definition finds it by `where: {params.joint.space_level, params.joint.band, params.joint.recording}`. One more preparation step to run before `run cortical_cells`; the alternative is a `setup` step that writes it. It is per recording
  (the name carries it), though the cells depend only on the time line: two recordings on one line have two joint Selections.
- **A row of the array is a spatial tile, not a vertex.** `measurement_values` reads [rows × members × comps]: here rows are the level's tiles, so the generic `statistic compute` over the joint Selection summarizes ACROSS time tiles
  per spatial tile (the time side's reading), not across space. A summary across space or across cells wants the cell to be the member (the code layout (tile << B) | time tile is the member axis the atlas uses); the
  array layout [S × T] and the code layout disagree by design (the code has gaps when F is not a power of two). 2b's events over cells need to read a cell by code: a helper `cell_index(code) = (tile, frame)` and its inverse.
- **Coarse levels are global; localization is tested where it must hold.** `test_cortical_cells` plants the level's OWN pattern (F_l δ at a tile's centre vertex, oscillating at 12 Hz in known time tiles, two tiles at different
  times) and finds the largest cell of each planted time tile in the planted tile. Measured on the two-hemisphere test surface (162 vertices a hemisphere), the pattern F_l² δ_c has its largest tile mean in its own tile for 12 of 16 tiles
  at level 3 (a few global modes: a neighbour's tile can win), 32 of 32 at level 4, 62 of 64 at level 5. The test runs at level 4. A bump WIDER than a tile (σ 0.8 m) is no one tile's: its largest cell is another tile. Phase 2b: events at a coarse
  level are placed to within the band's scale, not the tile; compare cells of one level.
- **No multirate.** The analytic signal is at the recording's rate; the Studio's chain reads the band every D samples (`chainStride`, about 30 samples a cycle) and nxr's analytic cross-spectrum estimator decimates. The runner has no
  decimating step (a chunk step's width is its input's); at 1200 Hz an alpha band is 100 samples a cycle, so the |z| accumulation is 3-4× the work it needs, and M_l x costs the same. Needs a `downsample` chunk step whose output is on a coarser
  time line (a new Value time axis), or the accumulator reading every D-th sample (the tile width stays a multiple of D).
- **F_l from the stored modes: BUILT (feat/eigenmode-first).**
  `transform: auto` is the default of `patches compute` and `apply_kernel`: the stored package's exact (Laplace–Beltrami) modes when they reach the band (λmax ≥ the band's top with its Meyer edge), else `eigen_solve` computes and stores them (a message in the progress,
  the report and the provenance), else Chebyshev, announced (`warnings`: level, the band's top, the modes' top, the resolved degree). `eigen` refuses where `auto` would fall back; `chebyshev` is as before; `exact` (dense) stays as the test reference, decided: it is the cheapest
  independent check of the other two and no cortex can use it (4096-vertex cap), so it is not folded into `eigen`. The glue map was not needed as a separate piece: the package carries it (`hodgeDirac_support` = the input vertices of the glued surface's first rows, the bridge after them), and the band is applied ON the glued surface
  (zero on the bridge, read back on the kept vertices, 0 on the cut wall), as `cyclepattern`/`phasegradient --spatial-band` apply theirs. That is not the same operator as the hemispheres' own pencil (which the series uses): near the medial wall the two differ; recorded in `spatial_band.package.domain`.
  The choice enters the cache key through a generic `StepSpec::resolve` hook (read-only, before the key): `key.params.resolved`. Issues for later:
  - **Cost of computing the modes**: the standard 3000-mode package of a 20k-vertex cortex is minutes to hours of one solve and ~240 MB of eigenvectors, stored for every later user. A level needs about 1.6 · 2^l · pieces exact modes (two hemispheres: level 8 ~800, level 9 ~1600); a 3000-mode package reaches level 9, finer levels are the series (announced) or a larger `modes` (cap 8000).
  - **Reading** a package reads the whole eigenvector file [V_glued × K] float32 once per band and keeps the band's columns (a 20k × 3000 package: 240 MB read per `apply_kernel` preparation); a column-wise read of only the band's modes would avoid it.
  - **The medial wall's vertices get 0** in the band (they are not on the glued surface); with the series they get the hemisphere's own band. For patches over the wall, or kernels with sources on it, this is a visible difference.
  - **Compute inside a run's preparation**: `apply_kernel`'s preparation may now write the package (the dataset is passed to `ExecContext`); a long solve is therefore inside `run`. If that is unwanted in a run, `eigen solve` first, or `--set transform=chebyshev`.
  - **The package is the first one stored**: `auto` never replaces a package that exists (other modes or mesh): a package that does not reach the band is the announced series, not a re-solve (`eigen solve --replace --modes N` does that deliberately).
- **What preparation resolves is recorded (fixed).** A step may report what its preparation resolved (`StepSpec::record`, generic): the runner puts it in the run's `prepared` (the report, the producer_json of Fields, spans and statistics,
  and the `recipes` dictionary entry that the measurement rows' `computed_by` names, which now exists for every measurement output of a run). `apply_kernel` reports its spatial band (level, member, λ range, transform, resolved order, series
  error, λmax). Tested (the order appears in the report and the recipe). The key still holds the request (order 0, the tolerance): the resolved order follows from them and the mesh.
- **Memory of `apply_kernel` with a band**: K [vertices × components × channels] doubles is built in full, then each component's [vertices × used channels] block once more (a 20k-vertex free cortex, 300 channels: 144 MB + 48 MB).
- **|z| is the mean of the magnitude, not the magnitude of the mean:** a cell's amplitude averages |z| per vertex (the sign and the phase within the tile are lost), as the time tiles' amplitude averages |z| per channel. A phase-aware
  cell measure (the tile's coherent sum) is a different measure, not built.
- **The time tile rule is all-or-nothing per time tile**, for every cell: a bad sample NaNs the whole time tile for every spatial tile, as the time side does.
- **Events (2b) and the graph**: a cell's neighbours are the tile graph of the level (same time tile) and the same tile in the next time tile; with F cells in time and a multi-hour recording, the union over chunks is the streaming
  question already logged (the labels cross the chunk edge); the cells' accumulators here are per run (the whole span), not per chunk, so 2b can read the stored cells after the run (the way `detect_spans` reads the time tiles).

## Global bursts: the graph (phase 2b step 1; the decisions are in "Global bursts: WHEN, from the cortex" above)

When the cortex, as a whole, is in a band's burst: the global time series of the band's activity, detected as time events (the rule of `band_bursts`) and fitted for their rise and fall (Rheome's `detect.risefall`). Built on the cells of phase 2a. Where
the activity is, and its extent, are phase 2b's later steps. Settled by the maintainers:
1. **Detection is on the time tiles of a TRACE**: per time tile, the MAX over the level's spatial tiles of the joint cell amplitude (a reduction over the spatial partition: the measure `max`). Whole brain by default; `per_hemisphere` is a named option.
   Then `detect_spans`, unchanged (Rheome's 1.5 / 1.0 SD, 0.1 s join, 0.4 s keep; the baseline a statistic, `summarize`, as `band_bursts`). Out: a spans Selection of GLOBAL bursts on the recording's time line (no per-channel rows).
2. **Edges refined on SAMPLES** (Rheome's design, coarse to fine): in the SAME chunk loop the trace is also made at the sample rate: per sample the area-weighted mean |z| per spatial tile (a spatial reduction, time unreduced), then the max over the
   tiles: a [groups × samples] series (Rheome's A). The [tiles × samples] intermediate is never stored (it lives inside one chunk step). The time tile replaces Rheome's 0.1 s moving mean for the DETECTION only.
3. **`risefall`**: a port of Rheome `+rheome/+detect/risefall.m`: the double sigmoid fitted to the sample trace over [t0 − margin, t1 + margin] of each burst: tOn, tOff, tauR, tauF, the baseline b and the height h, r², ok, the plateau. Per-burst measurements
   over the bursts Selection. labelalpha's r² ≥ 0.7 is a named option `min_r2`, default none (every burst kept, r² stored, the caller filters).
4. **Per-burst measures also**: the roll-up of the trace's tiles (amplitude, envelope_max), duration and interval (`rollup_events`, reused).

**`per_hemisphere` (my call, recorded).** Two traces (rows: one per connected piece of the surface, a cortex's hemispheres), `detect_spans` with `combine each` finds events per row, in ONE Selection whose events carry their row (`params_json.channels`, as
the per-channel bursts do; here the row is the hemisphere, named in `params_json.groups`). One set rather than two: the same output name, the same statistics code, the hemisphere is a property of the event. Whole brain: one row.

**The graph** (chunk loop over the recording; steps are the runner's; the nodes pure functions)

```
recording → analytic → apply_kernel (M_l = (F_l ⊗ I) K) → magnitude |z| per vertex            [phase 2a's chain, unchanged]
   ├─ measure over the joint cells → amplitude, envelope_max  [S × T]                           (accumulate)
   └─ tile_max (per sample: area-weighted mean per spatial tile, then the max over the tiles of each group) → [groups × t]   (chunk, pointwise in time)
        → collect (the own columns of every chunk appended: [groups × samples])                  (accumulate)
after the loop:
   reduce_cells (the max over each group's spatial tiles of the cells; the time level's meta) → trace_amplitude, trace_peak [groups × T]
   → summarize (median, mad)                  → baseline         (statistic)
   → detect_spans (1.5 / 1.0 SD, 0.1 s, 0.4 s) → bursts          (spans Selection on the time line)
   → rollup_events (trace_amplitude, trace_peak, duration, interval_next)                         per-burst measures + summarize across the bursts
   → risefall (the sample trace + the bursts)  → t_on, t_off, tau_rise, tau_fall, fit_baseline, fit_height, fit_r2, plateau   per-burst measures + summarize across the bursts
```

- **Types checked in the plan**: `tile_max` takes |z| on a surface with the tile ladder's level; `reduce_cells` takes a measurement over a joint partition (and refuses `per_hemisphere` on a surface of other than the pieces the ladder names); `detect_spans` and
  `rollup_events` take the reduced trace, whose `over` is the TIME level (the cells' time tile), so they are the unchanged steps; `risefall` takes the bursts and the sample trace of the same recording and groups.
- **Reduction rule (cardinal)**: the chain is pointwise; the spatial reductions are partition measurements (the area-weighted mean per tile: `nxr::partition`'s mean, over the tile partition, for every sample; the max over tiles: the measure `max` over the
  group partition); the time tile reduction is the `measure` accumulate over the joint cells. `detect_regions`-style detection is not used: time events are `detect_spans` on the trace.
- **Checked against the Studio's chain and `band_bursts`**: the same order (the linear maps before the first non-linear step, D106), the same time tile rule (whole and good), the same detection and baseline, the same storage of events and their measures.
- **What is held**: M_l as in phase 2a; per chunk the [tiles × chunk samples] means (inside `tile_max`), and across the run the [groups × samples] trace in double (one hour at 1200 Hz, two groups: 70 MB): stated in the step's comment; the fit
  reads each burst's window only.
- **Not built** (recorded, never a silent substitute): the `posterior` model of `risefall` (a grid posterior; refused by name), the trace at a decimated rate (the runner has no multirate: the trace is at the recording's rate), the trace stored as a Field.

## Global bursts build notes: issues for the next steps of phase 2b

- **The optimizer is not Rheome's.** `lsqcurvefit` (trust-region-reflective) is replaced by a bounded Levenberg–Marquardt with central finite differences. On the four synthetic episodes the two agree to 1e-5 s; on a badly posed episode
  (a flat or double-peaked window, tau at its bound) they may reach different local minima. The `atBound` flag is carried in the run's record, not stored per burst.
- **The `posterior` model is not ported** (a 4-D grid posterior, `GridMax`/`FramesMax` bounds, an autocorrelation-tempered likelihood). It is Rheome's best estimate under an ongoing background (MEG at 10 dB); refused by name. If the maintainers want it, it is a second port.
- **Bad samples are not excluded from the sample trace.** `collect` keeps every sample; the time tiles exclude bad ones (a tile with a bad sample is NaN and ends a burst), so no burst contains a bad sample, but the fit's margin (±1 s) can. Options: NaN the bad samples and let `risefall`
  interpolate or shorten its window; refuse a burst whose margin holds one.
- **The trace is at the recording's rate** (no multirate) and is HELD for the run (groups × samples doubles: one hour at 1200 Hz, two groups, 70 MB). A decimating step would cut both; Rheome fits at the envelope's rate.
- **Whole brain with `combine: mean` vs `each`**: the global events have row −1 (a fit on the mean over rows, valid on every row); per hemisphere the row is the hemisphere. Statistics across the bursts are per row (`combine each`): with one row they are the global ones.
- **The max over tiles is a noise-biased trace**: the max of 2^l noisy tile means rises with the number of tiles (a finer level has a higher noise floor). The robust baseline (median and MAD of the same trace) absorbs it for detection, but amplitudes are not comparable across levels. A tile-count-free
  trace (a quantile, a power-weighted mean) is a different measure.
- **Per-burst WHERE is not yet measured**: which tiles belong to a burst (the cells above the baseline in its time tiles, the patches of phase 1 at each time tile), its centre and extent: the next step, on the stored cells.
- **Runner**: `reduce_cells`, `tile_max`, `collect` and `risefall` are four more registry steps; the executor is unchanged; `StepSpec::record` reports nothing for them (nothing is resolved in preparation).

## Phase 2c: cells over many level pairs — the graph

Scope: the cells of phase 2a for a SET of (spatial level, time band) pairs in ONE pass over the recording: the joint plane per tile. Nothing new is measured: each pair is stored exactly as `cortical_cells` stores one, so every reader of
those rows works unchanged. Statistics, scale selection and events across pairs are later steps (the proposal's items 2-5).

**The decisions** (mine, recorded; the maintainers' rules apply)
1. **The caller declares two lists and the run takes their product**: `space_levels` (the surface ladder's levels, each its Weyl band) and `bands` (the time bands, each the band of one constant-Q level, D84). The pairs are
   every (level, band). A set that is not a product is two runs. (An explicit pair list would also be possible; the product is what "levels 3..7 × theta, alpha, beta" says, and keeps the work shared: one analytic signal a
   band, one bound map a level.)
2. **The run REQUIRES the joint Selections** (`joint compute`, one per pair and recording). The plan cannot create a Selection (it is pure), and a run that wrote them would hide a choice the maintainers made explicit (the joint
   partition is a stored node, made once). A missing one is refused BEFORE any data is read, naming EVERY missing pair with the command that makes it (the finder lists all missing inputs, not the first). A band that is
   no constant-Q level of the recording has no joint Selection, so it is refused the same way, the hint saying the band must be a level's `resolves_hz`.
3. **Repetition is the definition's, not a new step.** A definition may repeat an input, a block of steps or an output over the elements of list parameters: `for: {band: $bands}` with `do: [...]` (steps), or `for:` inside an
   input/output entry; the name and the values use the loop variable (`{band}` in a name, `$band` as a value). The runner expands the definition with the run's parameters BEFORE planning, so plan, executor, cache key and storage
   see the flat definition `cortical_cells` would be if written out by hand: the executor gains nothing analysis-specific. A name's `{band}` is the band as digits (`8_16`; a decimal point is `p`), `{level}` the integer.
4. **The graph** (per chunk, with the halo the band's taps need; `for band` outer, `for level` inner):

```
recording → for each band b:   analytic(b) → z_b                                                    ℝ → ℂ, once a band
              for each level l:  apply_kernel(z_b, M_l) → sources   (M_l = (F_l ⊗ I) K, bound ONCE a level, shared by every band)
                                 magnitude → |z|                                                    (non-linear: last)
                                 measure over joint(l, b): amplitude, envelope_max                  → [S_l × T_b] each, stored per pair
```

   Types are checked in the plan exactly as phase 2a's (the joint Selection names the same (space level, time level); its time level's band IS the band; contiguous from the first sample), once per pair. The executor frees a
   block after the last step that reads it, so one pair's `sources` and `|z|` are held at a time (not one a pair).
5. **Shared constants**: `apply_kernel`'s preparation reads the kernel, builds the level's spatial band and folds it into the kernel. The same (kernel, level, transform request) is the same constant for every band, so the
   run keeps ONE per key (a run-scoped store of bound constants in the step context: the Studio's "uploaded once"); the second band's step finds it. The spatial band's messages (an eigen solve, a fallback) are said once.
6. **Provenance and the cache key**: the key holds the parameters (both lists) and, per `apply_kernel` step, what its request RESOLVES to (`StepSpec::resolve`: eigen package or Chebyshev request), so a changed pair set or a
   changed stored package is another key; `prepared` (the resolved degree, series error, package) is recorded per step by `StepSpec::record` and names the pair in its key (`apply_kernel:sources_l5_b8_16`). A level's
   warnings are collected once.
7. **Storage**: each pair's two measures (rows of the recording, `amplitude` and `envelope_max`, and the `count` row) on that pair's joint Selection, the array [S_l × T_b × 2] beside the recording named
   `<recording>_cortical_cell_pairs.<measure>_l<level>_b<band>`. The rows are the same as `cortical_cells`' (a pair measured by either job is "made by another job" for the other: `--replace`, as always).

**What is held** (stated in the definition and the step comments, no limit set): the bound maps, Σ over the levels used of M_l [vertices × components × used channels] float64 (a 20k-vertex free cortex, 300 channels: 144 MB a level;
levels 3..7: 720 MB), held for the whole run; the accumulators, Σ over the pairs of cells × 2 doubles and a count (level 6 and 600 time tiles: 1.2 MB a pair); a chunk's work, one pair at a time: the block z_b [channels ×
(chunk + halo)] complex float32 a band while its pairs run, then one pair's `sources` [vertices × components × (chunk + halo)] complex and |z| [vertices × (chunk + halo)]; the per-chunk cost is bands × levels × (M_l z + |z| + two folds).

## Phase 2c build notes: issues for the next steps of phase 2b

Built: `run cortical_cell_pairs` (`pipelines/definitions/cortical_cell_pairs.yaml`), the definition repetition (`for:`, `expand` in `definition.h`), the executor's release of a block after its last reader, `ExecContext::shared` (the bound map once a level), the finder's list of every
missing input with its `hint`; test `test_cortical_cell_pairs`.

- **The pair set is a product, not a list.** A non-product set (fine levels only for fast bands, say) is two runs; a `pairs: [...]` parameter would need the analytic signal per distinct band and a per-pair loop (a `for` over a list of objects with a `distinct` rule). Not built.
- **The loops are the definition's, in order.** `for band` outer and `for level` inner is the written order; a block of steps nested the other way would hold every level's sources for a band at once. The release of a block after its last reader keeps one pair's sources at a time only
  because each pair's steps are written together.
- **Copies of a block**: the runner hands a step its inputs by value (a `Block` copy, as phase 2a), so a pair's sources are held twice for the moment of the magnitude. A step that took its inputs by reference would halve that; not changed here (the executor's contract is shared by every step).
- **Halos add**: the chunk's halo is the SUM of every chunk step's halo (the composition for a chain), so with several bands the sum of their taps is read round every chunk, not the longest. With three bands at 250 Hz this is a few hundred samples of 32768; a branching-aware halo (the longest path) is the exact form.
- **The bound maps are held for the run**: all levels' M_l at once (the sum is stated), even while one band's levels run, because the bands share them. A band-outer loop touches every level once a band, so they cannot be released between bands; a level-outer loop would need each band's z_b held. The trade is the sum.
- **Chunk independence is tested across loop chunks** (`test_cortical_cell_pairs_long`): the chunk service's loop chunk is the stored chunk, so a 40960-sample synthetic recording stored in 4096-sample chunks is ten loop chunks; two bands with different halos x two levels equal the direct computation on the whole recording at once, to 5e-8, including the time tiles at the edges (a bump is planted across one). (An earlier note here said the loop chunk was 2^15 samples: it is the stored chunk's width; 2^15 is the default only where the recording is stored so.)
- **The warnings and messages of a level are said once** (a level read by several bands gives one entry), but `prepared` and the key keep one entry a pair (the same resolution repeated a band times).
- **A time band that is two levels' band** (a level and its neighbour never share `resolves_hz`) is not a case; a band declared twice in `bands` would be refused by the repeated names.
- **Measures over an existing pair are another job's**: `cortical_cells` and `cortical_cell_pairs` store the same rows, so whichever ran second needs `--replace`; this is the stored-rows rule (one job's rows per Selection and measure), not a bug, but a pairs run over a set that overlaps earlier single runs is refused as a whole.

## Phase 2b: where per global burst — the graph

A global burst says WHEN (a span of time tiles). Its WHERE is the cells it covers at the run's (spatial level, time level) pair: the space × time EXTENT of the burst. The same rule as everywhere (hysteresis on a graph,
`nxr::regions::detectRegions`, unchanged); only the graph is the joint one. Built into `global_bursts` as steps AFTER its detection, in the same run: the cells and the bursts are in it (the runner has no input that reads a stored
measurement, and one more chain pass over the recording for stored cells would repeat the whole run).

**The graph** (after the chunk loop; the nodes pure functions of arrays; each burst on the cells of its own time tiles)

```
cell_amplitude [S × T]  ── summarize (median, mad per spatial tile across the time tiles; combine each) ─→ cell_baseline [S × 2]   (statistic: the per-tile BASELINE, stored)
bursts (spans on the time tiles) ┐
cell_amplitude, cell_baseline    ┴→ cell_extent   per burst, on the cells of its time tiles [first, stop) (and, per hemisphere, of the tiles of its row):
                                       standardize  z = (amplitude − median_s) / SD_s  per cell (SD_s the stored robust SD; a tile with no spread is left out and recorded)
                                       graph        cell (s, f) – cell (s', f) for each pair of touching tiles (tileAdjacency of the level), and (s, f) – (s, f+1)
                                       detect_regions (unchanged): hysteresis enter_sd / exit_sd on z, kept when ≥ min_cells cells and ≥ min_area (Σ tile areas of its cells)
                                     → labels   a labelling of the joint cells (a `set` Selection on the joint partition: code = the event's ordinal, −1 none)    (stored: Selection)
labels, cell_amplitude, cell_peak ─→ extent_measure  each event's measures by the exact roll-up of its cells (a measurement over the labelling)                    (stored: measurements)
                                       n_cells · n_tiles · area (the union of its spatial tiles) · amplitude (area and sample weighted mean) · envelope_max · first_tile · last_tile · duration
                                       · burst_ordinal (the burst it belongs to) · path_length (centres' chord, see below)
                                       and, per time tile, over the time level with the events as rows: area (Σ tile areas of its cells) · centre_vertex (the vertex nearest the area-weighted centroid)
                                     → summarize  the distributions across the events (count, sum, median, mean, sd, percentiles) of the scalar measures                  (stored: statistics)
```

**Decisions (mine; recorded for the maintainers, each changeable by a named option)**
1. **All regions of a burst are events, each linked to its burst by the burst's ordinal** (`burst_ordinal`). Two cortical patterns that do not touch in space-time within one burst are two events. Events are numbered by burst, then by their lowest cell
   (earliest time tile, then lowest spatial tile), so a rerun numbers them the same.
2. **The detection is confined to the burst's time tiles** (the burst says when): a region that would continue past the burst's last time tile is cut there. With `groups: per_hemisphere` the burst's row is its hemisphere and the cells are that hemisphere's
   tiles only (the same domain as its trace).
3. **The baseline is per spatial tile over ALL the run's time tiles** (each tile's own history, DECISION 3's first option; as `band_bursts` does per channel): median and robust SD (1.4826 MAD) by `summarize`, read by the detection, stored. The bursts are inside the
   history; the median and MAD are robust to them. A tile whose SD is 0 (no spread) cannot be standardized and is left out of the detection (its count is recorded), as `detect_spans` skips a series with no spread.
4. **Thresholds are Rheome's, as the bursts' own**: `enter_sd` 1.5 and `exit_sd` 1.0 SD above each tile's median (named options: `extent_enter_sd`, `extent_exit_sd`). Not a merge across gaps (the rule of space, DECISION 2); `min_cells` (default 1) and `min_area`
   (m², default 0) are named options, so no size limit is hidden.
5. **The cell code helper** (`nxr/extent.h`, `compute/`): code ↔ (tile, time tile) of the atlas layout (code = (tile << B) | frame, B = bitsFor(F)), with the checks (a frame beyond F, a tile beyond S refused). The labelling's array is the cell layout [S × T]
   (row = spatial tile, column = time tile: the layout of the cell measurements), so a cell's code and its place in the array agree through the helper; the codes of a closed form joint Selection are never stored.
6. **The labelling is array-backed** (an int32 array [S × T], the event's ordinal or −1): the cells of an event are in the array, never as JSON. Its `n_members` is the number of events; `params_json` names the bursts Selection (name and id), the pair (space level, time
   level, band), the thresholds and options, the baseline statistic, the per-burst result, and the cache key.
7. **Per-event measures are the exact roll-up of the event's cells** (the cardinal rule: nothing reduced by a node of the chain; the sums are over the labelling's members):
   - `n_cells` = number of cells; `n_tiles` = number of distinct spatial tiles; `area` = Σ areas of the distinct spatial tiles (the union; m²); the tile areas are the ladder's own (barycentric vertex areas summed by tile, the cells' weight);
   - `amplitude` = Σ (tile area × samples) × cell amplitude / Σ (tile area × samples) (the exact mean of |z| over the event's space-time support); `envelope_max` = the largest cell peak;
   - `first_tile`, `last_tile` (time tile indices of the event), `duration` = (last − first + 1) × the time tile's width; `burst_ordinal`;
   - per time tile (rows = the events, members = the time level's tiles; NaN where the event has no cell): `area` = Σ areas of its tiles in that time tile; `centre_vertex` = the vertex nearest the area-weighted centroid of its tiles' vertices in that time tile
     (`nxr::ladder::tileCentres`, a place: it does not roll up);
   - `path_length` = Σ over consecutive time tiles of the straight distance between the centre vertices (the chord; the surface's units, m): path dependent, not a roll-up (the spec's note on phase 1), so a function of the centres. A geodesic path is not built.
8. **Statistics across events** by `summarize` (descriptive only): count, sum, median, mean, sd, percentiles of `n_cells`, `area`, `amplitude`, `envelope_max`, `duration`, `path_length`. No nulls, no tests.
9. **Runner, kept generic**: a new output kind `labelling` (a `set` Selection on a product of the run, array-backed; stored with the spans and found in the cache the same way); a measurement over a labelling is over the run's events as a measurement over the run's spans is
   (the events' Selection is named by the measurement: `over.events_output`, no longer "the one spans output"); a statistic's rows axis names the spatial tiles when its rows are. The executor gains nothing analysis-specific.

**Types checked in the plan**: `cell_extent` takes x a measurement over a joint partition of the measure `amplitude`; baseline a statistic of the SAME measurement, combined `each`, with median and mad; events spans found on the TIME level of that joint partition (so the time tiles
agree); the surface and the level exist. `extent_measure` takes the labelling and, for amplitude and envelope_max, the cell measurement of that measure over the same joint partition.

**Checked against the Studio and nxr**: the atlas's joint partition (audit 1.2: code layout, closed form; here the labelling is its sparse instance); the Studio's `measureOverPartition` and nxr's `partition.accumulate` (the roll-up is sums folded into the member, merged by
sum and max); Rheome's `blobscale`/`labelalpha` for the thresholds. Events and tracks across time tiles belong to family C (tracking); this extent is a CONNECTED SET of cells, with the centre per time tile as its path, not a tracked object.

**What it cannot resolve** (in `producer_json.warnings` and `docs/reference.md`): the extent is a set of cells of one pair of levels: its size is quantized to the level's tiles and time tiles, and a pattern wider than a tile is placed to within the band's scale (phase 2a notes). A region is "above this
tile's own baseline", not "an oscillation propagating": speed and direction are measurements of the patterns' phases (phase 3), not of the cells' amplitude.

**What is held**: the cells [S × T] doubles and the labelling [S × T] int32 (level 6, 600 time tiles: 77 000 cells); per burst the window's graph (pairs × frames); the centres need a vertex pass per time tile of an event (20 000 vertices, hundreds of tiles).

**Tests (synthetic only):** a planted episode at a known tile moving to a neighbouring tile over consecutive time tiles inside one global burst, a second static episode elsewhere in another burst; the extents found with the right cells, the area and the path length against direct; a cell code round-trip;
a rerun is a hit; a replaced run; refusals.

## Phase 2b build notes: issues for what follows

Built: `nxr::extent` (`compute/include/nxr/extent.h`: the cell code, the space-time graph, standardize, windowRegions, rollup, areaByFrame, centres, pathLength; `test_extent`), the steps `cell_extent` and `extent_measure` (`pipelines/src/extent_steps.cpp`), the output kind `labelling`,
measures 1024 … 1028 (renumbered at the merge with phase 2d, which took 1022 and 1023), and the steps added to `global_bursts` (`test_burst_extent`). `detectRegions` and the executor are unchanged.

- **The thresholds are in SD of the tile's own history, and a coarse level's footprint is global.** At level 4 on the test cortex (162 vertices a hemisphere), one tile's own pattern has z of 3000 … 8000 SD in EVERY tile of its hemisphere (the test's noise SD is tiny against a planted oscillation).
  With Rheome's 1.5 / 1.0 SD an event is then the whole hemisphere during the burst; the test uses thresholds at 7000 / 4000 SD (about 85 % and 50 % of the peak) to localize. On real data the noise is not so small, but the footprint of a coarse band is still wide (phase 2a's note): the extent is placed to
  within the band's scale. Options: a threshold relative to the cell maximum within the burst (a fraction of the peak) as a named option; refuse or warn for levels below the one where the band is local (about 5 on a hemisphere-sized surface); report the footprint (n_tiles / tiles in the hemisphere).
- **Cells that cancel in the sum of two patterns.** The test's hand-over of tile A to tile B (both on for 2 s) split into two events at the thresholds first tried: the patterns are signed, so |z| per vertex of their sum dips where they have opposite sign, and the cell amplitude falls below the exit level for one time tile. An event that
  pauses for one time tile is two events (no merging across gaps, by decision). A named `merge_tiles` option (the time rule's merge, in whole time tiles) is the natural answer if the maintainers want it: it joins events of one burst that touch in space across a gap of at most N time tiles.
- **The statistic of a joint measurement had no valid rows axis.** `store_statistic` named the rows' axis after the recording's first factor (sensors); the baseline's rows are the level's spatial tiles. Fixed generically: the axis is `spatial_tile` with the level Selection when the measurement is over a joint partition.
- **A `set` Selection with an array needs a dictionary** (a model rule): the labelling stores one, `<labelling> events <hash of the cache key>`, one entry per event with its burst, row, first and last time tile and cell count. A replace with other options makes another dictionary (old ones are kept, never deleted).
- **The labelling's `params_json` must not use the key `joint`.** A first version recorded the pair under `joint` (as the joint Selection does); the input finder's `where: {params.joint.space_level: …}` then matched the labelling too, and the run's inputs changed after the first run (a rerun was no hit). It is `cells_of` now.
  The finder's `where` matches any `set` carrying the keys; a second Selection on the same product must keep clear of them.
- **A run with no events stores nothing for the event measures** (the existing rule: a measurement over zero members is not stored); the labelling is stored all −1 and a rerun is a hit (a measure whose members or rows are the run's events is found done when the events Selection has no member). The same had been
  true of `rollup_events` with no burst: a rerun was a miss and then refused as "exists"; now a hit.
- **Per-time-tile measures over the time level have the events as rows.** `area` and `centre_vertex` per (event, time tile) are measurements over the time level Selection whose ROWS are the labelling's events, not channels: the array has no axis metadata, so the reading is in the recipe and the step's meta (`rows_are`). Two
  runs with different labellings write the same (time level, recording, measure) key: the second is refused without `--replace`, as for any measurement. A cleaner form is a measurement over a partition whose members are (event, time tile) pairs, which the model has not got; the Studio keeps per-level arrays of explicit values (audit).
- **The path is the chord of nearest-vertex centres.** The centre jumps between vertices of a tile set as cells enter and leave; a static event's path is not 0 (1.28 against the moving event's 4.30 in the test). A smoother centre (an amplitude-weighted centroid, or the centroid itself projected nowhere) and a geodesic path would change the numbers; the centre is a PLACE, so no roll-up applies.
- **One burst, one baseline.** The baseline is over the whole run's time tiles including the bursts (a robust median and MAD). A run of one chunk (the default) uses that chunk's tiles only, so event extents on `--chunks 0` are relative to a short history; use `--chunks all` for events (the bursts need it too).
- **Memory and streaming.** The cells, the labelling and the per-burst graph are in memory after the loop (S × T doubles and int32; a burst's graph is pairs × its time tiles). For an hours-long recording the cells (level 6, 600 time tiles per 10 minutes) are small; the streaming-labels question of phase 2b's earlier notes does not arise
  here, as the extent reads the reduced cells after the loop, never the recording.
- **Decided by me, for the maintainers to change:** all regions of a burst are events; detection is confined to the burst's time tiles; the baseline per tile over all the run's time tiles; Rheome's thresholds as the extent's default; `min_cells` 1 and `min_area` 0 (no hidden limit); the centre per time tile by nearest vertex, the path by chord.
- **Not built:** merging across time gaps, geodesic path, tracking an event across bursts or levels (family C), the stored cells as an input of a separate definition (the runner has no measurement input), nulls and tests.


## Phase 2d: quadrature per cell and per burst — the graph

**Why.** Is the activity at a joint cell, or in a global burst, TRAVELLING or STANDING? A travelling or rotating pattern has a source cross-spectrum with a quadrature (imaginary) part: Im T is what no zero-delay mixing makes
(`nxr/quadrature.h`: a real map applied to the sensors turns Im C into B (Im C) Bᵀ, adding none of its own). A standing pattern is one real pattern times one phase: Im T = 0. The reading is per spatial tile s of a level l
(the level IS the spatial band) and per time band at its constant-Q level, so each tile of the joint plane, and each burst, carries its own answer, from stored products only (item 3 of the joint-plane proposal, "What is
missing"). No recording pass: every input is a stored node.

**The quantity, and why it is stored as ingredients (the cardinal rule).** For a time tile t the stored sensor Σ z zᴴ is C_t (`crossspectrum compute --estimator analytic`, a statistic with a window axis). For a spatial tile s
of level l, B_s are the rows of M_l = F_l K of the tile's vertices (ALL components of a free kernel; no orientation assumed; M_l real). The tile's source cross-spectrum is **T = B_s C B_sᵀ** (rows = the tile's vertices ×
components). The measure is the ratio ‖Im T‖_F / ‖T‖_F (nxr's `lagged_fraction`, the one `quadrature compute` stores per band). It is a RATIO of two sums, so:
- **stored** (sums of products, the only things reduced): `cross_spectrum_norm2` = Σ_ij |T_ij|² = ‖T‖_F² and `cross_spectrum_quadrature_norm2` = Σ_ij |Im T_ij|² = ‖Im T‖_F² (new nxr-neuro measures, codes 1022 and 1023, reduction
  sum, registered in `io/schema/measures_nxr.json` as the phase-gradient family is);
- **derived at read** (never stored, never reduced): `quadrature_fraction` (1111) = √(‖Im T‖² / ‖T‖²). It lies in [0, 1/√2 ≈ 0.707] for ONE pattern (a rank-one T = u uᴴ, u = r + i j: ‖Im T‖² = 2(|r|²|j|² − (r·j)²) ≤ ‖T‖²/2),
  up to 1 for several patterns at different phases; 0 for any standing pattern. The squares are stored (not their square roots) because sums of squares add and norms do not.
- The cell's own T is the tile's block of the source cross-spectrum: ‖T‖² counts Σ over the tile's PAIRS of vertices and components, not the sum of per-vertex powers.

**Without forming T.** B_s is [n_s·components × channels], T is [n_s·components]²: dense it is too large to hold per cell. With the channel Gram G = B_sᵀ B_s (channels × channels, real, per spatial tile, once):
‖B X Bᵀ‖_F² = tr(X G Xᵀ G) for any real X, and B real, so with C = Cr + i Cq (Hermitian: Cr symmetric, Cq antisymmetric) ‖Re T‖² = tr(Cr G Cr G) = Σ_ij (Cr G)_ij (Cr G)_ji and ‖Im T‖² = −tr(Cq G Cq G); ‖T‖² =
‖Re T‖² + ‖Im T‖². Two channels × channels products per (tile, time tile) and nothing larger. C is symmetrised ((C + Cᴴ)/2, a no-op to round-off for a stored sum of z zᴴ) before. Tested against the dense T on a small tile.

**The roll-ups, stated (which are exact).** T is a quadratic form in C; ‖·‖² is not additive over blocks:
- **Over TIME, exact only in C.** C summed over a burst's time tiles is the burst's own cross-spectrum (products add): T_burst = B (Σ_t C_t) Bᵀ. But Σ_t ‖T_t‖² ≠ ‖T_burst‖² (the cross terms between tiles are in the second, not the first). So the
  per-BURST quantity is NEVER the roll-up of the per-cell ingredients over time: it is made from the summed C (below).
- **Over SPACE, not exact.** Σ_s ‖T_s‖² omits the cross-tile blocks of the union's T, so the ingredients of a union of tiles are not the sum of the tiles'. They sum to the energy of the block-diagonal of the source cross-spectrum (the
  within-tile blocks): a defined quantity, never the union's.
- **What the per-cell sum over time means** (the stored rows' roll-up by `sum`): the energy of the block-diagonal in time and space; its derived fraction is the power-weighted mean of the cells' own readings, never the fraction of a union.
The stored measures say so in their descriptions. No roll-up claims more.

**Per BURST.** For each burst (a spans Selection of global bursts on the recording's time line), its time tiles are the stored tiles WHOLLY inside [start, stop] (the tile edges, as `detect_spans` made them: the bursts are tile-aligned);
C_b = Σ_{t ∈ burst} C_t (the partition's sum over its members, `nxr::partition`); then for each spatial tile the two ingredients of B_s C_b B_sᵀ. A burst whose tiles are not all stored (a tile dropped for a bad sample or the span's end)
reads NaN and is reported, never a partial sum. Stored as measurements over the bursts Selection: an array beside the recording [spatial tiles × bursts × 2] (rows = spatial tiles of the level, members = the Selection's events in order,
components = the two ingredients), the same form `rollup_events` stores.

**The graph** (no recording pass; one verb, `cellquadrature compute`, as `patches compute` composes pure nodes; the executor is the runner's no-chunk path: there is none, the precedent is a verb)

```
LOADING (io, by name)         cross-spectrum statistic <stem>_<band>_windows   (Σ z zᴴ per time tile, on the kernel's channels, in its order)         ℂ [W × m × m]
                              kernel (free or fixed), the surface's tile ladder, the joint Selection of the pair, (optional) the bursts Selection
PREPARATION (host)            type checks: cross-spectrum tile level = the joint pair's time level (window_samples = 2^level), same recording; kernel from the cross-spectrum's channels, to the joint's surface;
                              the joint's space level exists; bursts on the same recording's line. M_l = (F_l ⊗ I) K bound ONCE (spatial_band.h: eigenmode first, Chebyshev an announced fallback)       ℝ [rows × m]
GRAPH (compute/, pure)
  cells:   congruence_norms   per spatial tile s, per time tile t:  (‖T_st‖², ‖Im T_st‖²) from (B_s, C_t), via tr(C G C G)                                        → ℝ [S × W] ×2
  bursts:  sum_over_members   Σ_{t ∈ burst} C_t  (the time partition's reduction of products)                                                                   → ℂ [B × m × m]
           congruence_norms   per spatial tile s, per burst b:  (‖T_sb‖², ‖Im T_sb‖²)                                                                            → ℝ [S × B] ×2
READBACK (io)                 measurements over the joint cells (array beside the recording [S × F × 2], F the joint's time tiles, NaN where the tile is not stored)
                              and over the bursts [S × B × 2]; computed_by = the pipeline and its key; the fraction is DERIVED at read (`quadrature_fraction`)
```

- **Checked against the paramount architecture.** Compute: `quadrature::groupedCongruenceNorms` (a function of matrices, no reading, no tile or burst knowledge: a real matrix, its row groups, a list of Hermitian matrices) and
  `partition::sumOverMembers`. The reductions are partition reductions: the burst sum is a sum over the members of the time partition, the ingredients are sums of products. Orchestration reads stored nodes, type-checks, binds M_l once,
  hands the arrays over and stores; it does no maths and no loop over samples or chunks (a loop over the stored time tiles is the partition's members, inside the node). Data (io) reads the cross-spectrum by name; there is no recording.
- **The Studio's chain**: the linear map (kernel with F_l folded in, D106) is applied to the stored sums of products, not to samples: K (Σ z zᴴ) Kᵀ = Σ (Kz)(Kz)ᴴ, the same ordering (linear before non-linear); the
  quadrature is the first non-linear (quadratic) step, read from products. The Studio has no quadrature chain; nxr's `lagged.py` (`im_fraction`) is the reference and `quadrature compute` the sensor-to-source analogue.
- **Cache and provenance.** The key holds the cross-spectrum (id, modified), the kernel, the joint Selection, the bursts Selection (id, modified, its tiles), the spatial band's plan (`plan_spatial_band(...).key`, so the choice of
  eigenmodes or series is in it), the compute version; rows' `computed_by` names the pipeline and a hash of the key. A rerun is a hit; other rows of the same Selection and measure are refused unless `--replace`.
- **What it cannot show**: a standing pattern and a travelling one differ only in the quadrature of T, but T comes from a source estimate of a limited resolution: a source-space quadrature is the sensors' quadrature through K (the
  zero-delay mixing adds none), and the kernel's spread blurs a pattern into its neighbours' tiles. A cell reading is the fraction of that tile's own cross-spectrum only (its internal quadrature); the quadrature BETWEEN two tiles
  (a wave running from one tile into the next) is in the cross-tile blocks this measure leaves out (issues below). A tile's fraction is a summary of ‖Im T‖²/‖T‖²: it discards which pairs of its vertices lead which, the
  direction and the speed; those are `phasegradient` / `speedspectrum`, and the full per-pair Im T is `quadrature compute`'s pairs.
- **Not built, by decision**: statistics and nulls (a fraction against what a standing pattern with noise would read), the fraction over all the cortex's vertices (the full T over the whole surface), per-hemisphere or per-pair summaries,
  weighting by tile vertex area (the tile's vertices count equally).

## Phase 2d build notes: issues for what follows

Built: `cellquadrature compute` (`pipelines/src/cellquadrature.cpp`), `quadrature::congruenceNorms` / `groupedCongruenceNorms`, `partition::sumOverMembers`, the measures 1022, 1023 (stored) and 1111 (derived), `derived_measurement_values` (io), test `test_cellquadrature`. As the graph says; the decisions
that were left open:
- **Ingredients chosen: Σ|T_ij|² and Σ|Im T_ij|²** (the Frobenius squares, not the Re/Im pair), so the fraction is nxr's ‖Im T‖/‖T‖ (`lagged_fraction`) and the squares add where a sum is wanted. `Re T` is `‖T‖² − ‖Im T‖²` at read if wanted.
- **A verb, not a definition.** The runner's executor is a chunk loop over a recording; this pipeline has no recording pass, so the precedent is `patches compute`: a verb composing pure nodes over stored rows. Nothing in the executor changed.
- **The joint is found, not named, by default:** (recording, space level, the cross-spectrum's tile level), since a constant-Q level is one distinct window (D84). `--joint` names it. A cross-spectrum on another tile level than any joint of the recording is refused with the levels there are.
- **The cache key does not hold the joint's or the bursts' `modified_utc`**: writing measurements over a Selection stamps its modified time, which would make every rerun a miss. The joint is identified by its id and its layout's hash, the bursts by their id and a hash of their events.
- **Array dtype is float64**, unlike the cells' float32: squares of MEG products fall below float32's range.
- **Per-burst WHERE (the bursts' spatial extent) is not built**; this gives per burst and spatial tile the quadrature reading, which together with the burst's per-tile power (`cross_spectrum_norm2`) says where the burst's energy is travelling or standing.
Issues:
- **Noise reads a high fraction.** On a short window a noise-only tile reads about 0.3 (a window of a band of width B and length T holds about B·T independent samples), against 0 for a standing and up to 0.707 for one travelling pattern. A fraction is only to be read where the tile carries power; no null is built (out of scope). A burst, summing its tiles, has more degrees of freedom than a cell.
- **Within-tile only: now answered (2026-11-13, below).** The quadrature between neighbouring tiles is stored as the tile-pair ingredients (`tile_pair_*`, 1029-1030).
- **Cost.** 2 m³ flops per (spatial tile, time tile): level 6 (128 tiles), 600 time tiles, 270 channels about 10 minutes on a workstation. A tile with fewer rows than channels could use its thin factor (Y = Σ Vᵀ, r < m) for r m² + r² m; not built. The tiles of C are all read at once (600 × m² complex: 700 MB).
- **Unweighted by area: now the named option `weights: none` (2026-11-13, below).** The default is `area`.
- **The cross-spectrum's channels are the kernel's** (read on the kernel's channels in its order); a cross-spectrum on other sensors than the kernel's refuses. The same cross-spectrum read through two kernels is two jobs on the same rows (refused unless `--replace`): the rows' key has no kernel, as `measure compute`'s has no recipe.
- **Per-hemisphere bursts** (two events overlapping in time, one per hemisphere) each sum their own tiles; the quadrature of one event is read over ALL tiles of the level (both hemispheres), not the hemisphere's own.
- **The old Operator form of a cross-spectrum series** (before 2026-11-13) is not read here.

## Spatial analytic signal (monogenic): the graph

**Why.** A spatial band leaves a SIGNED field (lobes of both signs). Measuring |z| of it is the spatial analogue of measuring a band-passed x(t) instead of its envelope: where lobes of opposite sign meet, the field passes through zero and |z| dips
(the phase 2b "one-time-tile dips split events" issue: a pattern handing over from one tile to the next reads as two events). The spatial analogue of the analytic signal is the MONOGENIC signal (Felsberg and Sommer 2001): with the Riesz
transform R f = ∇ Δ^{-1/2} f (a tangent vector field), the spatial amplitude A = √(f² + |R f|²) is unsigned and has no zero-crossing dips; the spatial phase is atan2(|R f|, f) in [0, π]; the orientation R f / |R f| is the local wave-vector direction (a tangent vector).

**The maths on a surface (eigenmode first).** With the band's mode coefficients c_k = ⟨ψ_k, f⟩ (the vertex-mass inner product), f = Σ g_k c_k ψ_k and R f = Σ g_k c_k ∇ψ_k / √λ_k: the exact 1-forms dψ_k of the Hodge–Dirac pairing, as tangent
vectors per vertex by the mesh's P1 gradient (`phase::vertexGradient`, the area-weighted mean of the incident faces' gradients, as `phasegradient`). The constant mode (λ = 0) has no Riesz part (gain 0). By linearity R f = ∇(Δ^{-1/2} F_l f): the
band's gain g is replaced by g_R(λ) = g(λ)/√λ (0 at λ = 0), then ONE gradient. So the Chebyshev fallback is the same series with that gain (its own error checked against the tolerance; a scaling member, whose gain is not zero at λ = 0, has a
singular g_R and is refused, never approximated), announced exactly as the band's (eigenmode first, Chebyshev second, never silent).

**Space × time.** z(v, t) is complex in time (the band's analytic signal) and band-passed in space. R is linear and real, so R z = R Re z + i R Im z, and A² = Σ_a |z_a|² + Σ_a |R z_a|², summed over the source components a (a free kernel: per
component, then summed; no orientation assumed, as `magnitude`).

**The graph** (the chain stays pointwise; nothing is reduced):
```
LOADING (io)            recording blocks (chunk service); kernel K [3V·c-rows × channels]; the surface's tile ladder (and its stored eigenmodes)
PREPARATION (host)      type checks (the kernel's surface, the ladder, the level); the band F_l by the spatial band (eigenmode first); the Riesz gain g_R = g/√λ; the mesh gradient G (P1, on the faces whose vertices are all kept)
                        BOUND ONCE:  M_l = (F_l ⊗ I) K  [c·V × m]   and   R_l = (G Φ diag(g_R) Φᵀ ⋆0 ⊗ I) K  [3c·V × m]   (stacked per vertex: [M_l ; R_l], 4c rows a vertex)
GRAPH (pure nodes)      analytic(band)  →  apply_kernel(tiles, level, envelope: spatiotemporal)  →  [z ; R z] ℂ (field type "monogenic": 4c rows a vertex)
                        →  magnitude  →  A [V × samples] ℝ   = √(Σ|z|² + Σ|R z|²)      (compute: monogenic::amplitude; the same node `magnitude`, now the norm of the monogenic field)
                        →  measure over the joint cells (amplitude, envelope_max) …  (unchanged)
READBACK (io)           as before; the recipe's `parameters.envelope` and the step's resolved band record which envelope
```
- **Nodes (compute, `nxr::monogenic`, pure, called by tests directly):** `rieszGain` (g → g/√λ); `rieszTransform` (G, Δ^{-1/2}F_l f → R f); `amplitude` (z, R z → A); `phase` (f, R f → atan2(|R f|, f)); `orientation` (R f → unit tangent vector, NaN where
  |R f| = 0). Phase and orientation are nodes with typed ports in compute, tested, and NOT stored by default (no registry output until a pipeline asks for them).
- **Option `envelope: temporal | spatiotemporal`** on `cortical_cells`, `cortical_cell_pairs`, `global_bursts` (the cell amplitudes that `cell_extent` reads are these). DEFAULT `temporal`: the chain is byte-for-byte the old one. It is a parameter of the
  step `apply_kernel` (with `tiles`/`level`), so it is in the definition's parameters, the cache key and the producer record.
- **Rows of the two envelopes.** The stored rows' measures stay `amplitude` / `envelope_max` on the pair's joint Selection, and the unique key of a stored measurement does not hold the recipe, so the two envelopes of one pair cannot coexist as rows. They never
  collide silently: the cache key differs (`parameters.envelope`), so a run of the other envelope is a miss and the existing rows are refused unless `--replace` (the runner's rule for any recipe change), and `producer_json.parameters.envelope` records which
  one the rows are. Holding both side by side needs a measure name of its own (reserved range 1000–1999); that is the maintainers' decision (open item below).
- **Memory.** M_l and R_l are stacked: 4c rows a vertex instead of c (a 20k-vertex free cortex, 3 components, 300 channels: 12 × 20000 × 300 × 8 B ≈ 576 MB a level instead of 144 MB), shared by every band reading the level; a chunk's stacked sources are
  [4c·V × (chunk + halo)] complex float32 until `magnitude` releases them.
- **Checked against the architecture.** Compute: matrices in, matrices out, no file, no chunk, no choice of input. Preparation: type checks (`magnitude` accepts the monogenic field only from `apply_kernel`'s declared type), the maps bound once; D106: the
  linear maps (band, Riesz, kernel) all read the chain before the first non-linear step (`magnitude`). No reduction anywhere (A is per vertex per sample; the cells' amplitude is the measurement over the partition as before). The Studio's chain has no monogenic
  stage: this is a new linear-map member of the "members → filter → transform" order, before any non-linear transform.
- **What it cannot resolve.** The Riesz part of a field known only on the cortical mesh is the mesh's discrete gradient of Δ^{-1/2} f: near the cut medial wall (the glued band is zero on it) the gradient is taken on the faces whose vertices are all kept, so a
  vertex of the wall's edge has a one-sided gradient; A is trusted away from it. A band of a few lobes on a mesh of coarse resolution has its |R f| underestimated by the P1 gradient's smoothing. A is a local amplitude of the BAND's pattern, not of an event: an
  event's extent still comes from `cell_extent` on A's cell means.

## Spatial analytic signal build notes: issues for what follows

Built: `compute/include/nxr/monogenic.h` (`rieszGain`, `rieszTransform`, `amplitude`, `phase`, `orientation`), `SpatialBandOptions::riesz` and `SpatialBand::apply_riesz` (`spatial_band.cpp`, the three transforms: exact, Chebyshev, stored eigenmodes), `apply_kernel`'s parameter `envelope`
(field type `monogenic`, 4c rows a vertex), `magnitude` reading it, `envelope: temporal` in the three definitions, tests `test_monogenic`, `test_spatial_band` (Riesz part), `test_spatial_analytic`.
Decisions made:
- **No new step.** The spec's graph names `magnitude` as the node that makes A; it is the same step, now the norm over a typed monogenic field (checked by type: `magnitude` of any other chain is unchanged). The definitions need no conditional step: the option is a parameter
  of `apply_kernel`, in the definition's parameters, the cache key and the producer record. A change of an existing definition's text changes its hash, so a stored result of the old key is a miss (refused unless `--replace`), never silently replaced.
- **Rows.** The two envelopes' rows of one pair are not side by side (unique key without the recipe); a different envelope is refused without `--replace`. A measure name per envelope (`amplitude` / `envelope_max` in the reserved range) would hold both: the maintainers' decision.
- **The Riesz gain's tolerance is relative** to the peak of `g/√λ` (it is in metres; the band's gain is not), and the constant mode's cut is `λ ≤ 1e-3` 1/m² (no cortex has a non-constant mode there).
- **Phase and orientation are not stored.** They are pure nodes with tests; a registry step (a Field of the spatial phase, a vector field of orientations) is the next step if a pipeline reads them.
Issues:
- **The hand-over dip is reduced, not removed.** At the hand-over the cells' trough relative to the plateau is 0.60 (temporal) and 0.67 (spatiotemporal): the amplitude of a SUM of two patterns in the same time phase is the sum's, and where the patterns cancel in the field
  both `|f|` and `|R f|` can fall. The event no longer splits at an exit level that splits it under `temporal` (5909 vs 6988 in extent z), a margin, not a guarantee.
- **Cost.** Four times the rows of the bound map and of a chunk's sources (a free kernel: 12 rows a vertex instead of 3), and the Riesz part's own band (a second transform of every kernel column). Not measured on a real cortex.
- **Wall edge.** The gradient near the cut medial wall is one-sided (faces with all vertices kept); a cortex with a large wall has its A near the wall less trusted. A glued-surface gradient (on the package's glued mesh) is not built.
- **A level-0 (scaling) Chebyshev Riesz part is refused**, its gain not being 0 at λ = 0; `auto` there falls to eigenmodes when stored, otherwise the refusal says so.


## Phase 2e: scale selection — the graph

**Why.** An event has its OWN size and rhythm; we do not impose them (joint-plane proposal, section 5 and "What is missing", item 2). Detected over a declared set of (spatial level, time level) pairs, the same event is seen at several
pairs; the pair of its strongest response is its characteristic size (the level's tile, its Weyl band) and rhythm (the time level's band), and its response across the pairs is its footprint in the plane: a single-scale rhythm, or spread
over scales; coarse at onset and focal later, or the reverse. Rheome's `detect.blobscale` is the reference (read: `+rheome/+detect/blobscale.m`): the filter bank's response is made COMPARABLE across scales by a scale
normalisation (the mexican hat carries the factor t), the selected scale is the one of the largest response, READ AT THE BLOB (not a map's mean: the tails select coarser scales), and `atFloor` marks a selection sitting on the finest
scale of the ladder as unresolved. All three are ported; the comparable unit is ours (below).

**What the runner can and cannot express (checked, as the task asks).** The per-pair half CAN: `for:` repeats `cortical_cell_pairs`' chain over the pair set, and after the loop each pair's trace, its global bursts (`reduce_cells`,
`summarize`, `detect_spans`) and its extents (`cell_extent`) are the existing steps, unchanged, written once per pair: a new definition `cortical_event_pairs` (no executor change). The cross-pair half CANNOT, without a change to the
runner: a step has fixed named ports and the pair set is a list of any length (a step over N labellings and N cell measurements would need list-valued ports through the plan, the executor's input gathering and the cache key), and its
outputs are not a joint partition's. The precedent for a pure composition over STORED rows with no recording pass is a verb (`patches compute`, `cellquadrature compute`): `scaleselection compute`. The executor gains nothing analysis-specific;
the one generic fix in the runner is that a labelling names the bursts it was found against by its own step's input (it took "the" spans output of the run, which is ambiguous with one per pair).

**The graph** (per pair p = (l, m): spatial level l, time level m; P pairs = `space_levels` × `time_levels`)

```
RUN   cortical_event_pairs   (one chunk pass; for each pair)
        cells (amplitude, envelope_max over joint(l, m))  → trace per time tile (the max over the level's tiles; reduce_cells) → baseline (median, mad) → detect_spans → bursts_p        the pair's OWN global bursts
        cells' amplitude → cell_baseline_p (median, mad per spatial tile)  ;  cell_extent(cells, cell_baseline_p, bursts_p) → extents_p (labelling of the joint cells)
VERB  scaleselection compute   (stored rows only)
LOADING     per pair: the joint Selection, the labelling extents_p [S_l × T_m] int32, the cells' amplitude [S_l × T_m] and its baseline statistic (median, mad per tile)            (io, by name)
PREPARATION type checks: one recording and surface; each pair's labelling is of that pair's joint partition, and its baseline of the same cells; the pair set a product with every pair stored; z_p = standardize(amplitude_p,
            median_p, SD_p) (`nxr::extent::standardize`, the one the extents were found on)                                                                                      ℝ [S_l × T_m] a pair
GRAPH (pure)  event_facts   per event of a pair: its cells, its peak z, its mean z, its first and last sample
              link_events   events of different pairs are LINKED when ≥ `min_link` pairs of their cells nest in space and overlap in time (exact, below); the links' connected components are the cross-pair events
              select_scale  per cross-pair event: the response at every pair (the footprint), the best pair (the largest), the response there, the edge flags
READBACK    a `spans` Selection of the cross-pair events (their span in time) and the pair table in its params_json; measurements over it; the cross-pair event of each pair's event over the pair's labelling
```

**Decisions** (mine, each a named option where it is genuinely the caller's; the maintainers' rules apply)
1. **The comparable unit is the per-tile STANDARDIZED amplitude** z = (amplitude − the tile's median over the run's time tiles) / its robust SD, as `cell_extent` standardizes. A raw amplitude, or the raw max over tiles, rises with
   the number of tiles and with the band's noise (phase 2b's notes: "the max over tiles is a noise-biased trace"); z puts every pair's response in units of that tile's own history at that pair, which is also the scale normalisation that
   `blobscale` needs (an un-normalised response is always largest at one end of the ladder). It is NOT free of selection bias: a pair with more cells has more chances for a large noise z (issues below). Stated in `producer_json.warnings`.
2. **The event's response at a pair is the PEAK z over its cells** (`response: peak`, the default: "best pair = the pair of largest peak standardized response", blob-detection scale selection read at the blob). `response: mean` (the mean z over the
   event's cells) is the named alternative, less sensitive to one cell and to the number of cells; both are stored in the event facts, one is chosen for the selection.
3. **Matching across pairs by the ladders' nesting, exact, no distances.** A cell is (spatial tile s of level l, time tile f of a window of W samples starting at the pair's origin). Cell (s, f) at pair a and (s', f') at pair b NEST AND
   OVERLAP when: the finer level's tile shifted right by the level difference is the coarser level's tile (tile t of level l has children 2t, 2t+1, a tile's code is h·2^l + t, so s >> (l − l') is its ancestor at l'; equal levels: the same tile),
   and the two sample intervals [origin + f·W, origin + (f+1)·W) intersect (dyadic windows nest likewise; the test is the intervals'). Two events of different pairs are LINKED when at least `min_link` (default 1) pairs of their cells
   nest and overlap; the cross-pair events are the connected components of the links (the same rule as a region: connected nodes, here events). Events of one pair never link directly; they can share a component through another pair's event.
   The global burst is not a condition: events of one burst overlap in time, and the burst is a property of the pair (its trace), so events of the same episode seen at different pairs (each pair has its own bursts) are matched by their cells.
   Cross-pair events are numbered by their earliest sample, then their lowest member (pair, ordinal): a rerun numbers them the same.
4. **The footprint** is the matrix response [pairs × events]: the response of the cross-pair event's member events at that pair (the largest, when it has several there), NaN where it has none. Stored as a measurement `scale_response` whose rows are
   the pairs and members the events (the form of the cells: rows a partition's members, the table of the pairs is in the Selection's record). A pair where no event of the cross-pair event was detected is NaN, not 0 and not the sub-threshold z
   there: the footprint is a footprint of DETECTED events (the sub-threshold response at a pair is the cells' own measurement; reading it per event is not built, issues below).
5. **Best pair** = the pair of the largest footprint value; a tie goes to the first pair in the declared order (space level ascending, then time level ascending). Its spatial level, time level, and what they mean physically: the level's mean tile
   area (m²: the surface's area over the tiles of the level), the time level's band (Hz, lo and hi). Edge flags (`scale_edge`, a bit mask, set only for a dimension with at least two values): 1 finest spatial level of the set, 2 coarsest, 4 shortest
   window (the fastest band), 8 longest window: the selection sits on the edge of the set, the true scale may lie beyond it (`blobscale`'s `atFloor`).
6. **The cross-pair events live in ONE `spans` Selection** on the recording's time line, `<recording>_scale_selection` by default (`--name`): one event per cross-pair event, its span the union of its members' time tiles. (First written as a closed-form `set`
   on the joint product, as the joint partition; the `set` has no natural meaning for objects that are not cells of one partition, while a `spans` Selection is the existing form of "an event in time" with its measurements and readers, as the bursts.) The pairs
   (level, time level, band, tile area, the joint and labelling of each) and the options are in its `params_json.scale`. A cross-pair event's members are NOT in JSON: the link is the measurement `scale_event` over each pair's own labelling (the cross-pair event of each event).
7. **Stored measures** (registry 1031 … 1040, all descriptive, none rolls up): over the cross-pair events (one array [1 × E × 8], a component each): `scale_space_level`, `scale_time_level`, `scale_peak` (the response at the best pair, SD),
   `scale_tile_area` (m²), `scale_band_lo`, `scale_band_hi` (Hz), `scale_pairs` (how many pairs hold a member), `scale_edge` (the span in time is the Selection's own); the footprint `scale_response`
   [pairs × E]; over each pair's labelling `scale_event` [1 × events of the pair]. A replaced run rewrites them (`--replace`); another job's rows are refused.
8. **The pair set is a product** (`space_levels` × `time_levels`), as `cortical_cell_pairs`; every pair needs its stored cells, baseline and labelling (made by one `cortical_event_pairs` run over the same product): a missing one is refused
   naming EVERY missing pair and the command. The verb reads only; the run is the one pass over the recording.
9. **Not built, by decision**: statistics and nulls; the sub-threshold response of a pair with no event; a response as a function of the event's cells' union area (the area-weighted amplitude, `extent_measure`'s); a scale per time tile
   (a size that changes within an event: "coarse at onset and focal later" is read from the footprint per pair and the members' first and last tile, not tracked); a continuous scale (the maximum of a fitted curve over the pair axis).

**Types checked** (before anything is computed): one recording, one surface; every pair has its joint Selection, labelling (of that joint partition, cell layout [S × T] int32), cell amplitude and baseline (median and mad, over that joint partition);
the labelling's pipeline is the `--source` (default `cortical_event_pairs`); a level's tiles equal the joint's; no two pairs the same (level, window).

**Checked against the paramount architecture.** Compute (`nxr::scale`, `compute/`): pure functions of arrays (z, labels, level, window); they never read a row, never know a recording, a Selection or a file, never loop over samples or chunks, and reduce
nothing across time or space on their own (the peak and mean of an event are the labelling's members' sums and max, as `extent::rollup`). Data (io): the verb reads stored rows by name and writes Selections and measurements. Orchestration: the verb
composes them (no maths, no loop over samples); the definition composes existing steps with `for:`. The chain is a graph of pure nodes; the per-pair part is the Studio's cell chain repeated per pair, as phase 2c.

**What it cannot resolve** (in `producer_json.warnings` and `docs/reference.md`): the size and rhythm are quantized to the pair set's levels and bands (the best pair is a member of the declared set, not a fitted scale); a coarse level's footprint is wide
(phase 2a's note) and a band's time tile is longer at slow bands, so a tie of response across pairs is possible and is resolved by the order; the selection can sit on the set's edge (flagged); a response is a standardized amplitude, not a
measure of the pattern's shape.

**Tests (synthetic only):** two episodes in the two hemispheres at different times, A the pattern of a coarse level's tile at 6 Hz, B the pattern of a finer level's tile at 12 Hz, over two spatial levels × two time bands: the scale selection gives
A its level and band, B its own; the footprint peaks there and is NaN or lower elsewhere; the events found at several pairs are one cross-pair event each; the compute functions on hand-made pairs (nesting, overlap, `min_link`, ties, edges, `mean`); a rerun is a hit;
a replaced run; refusals (a missing pair, a mixed source, a non-product).


## Phase 2e build notes: issues for what follows

Built: `nxr::scale` (`compute/include/nxr/scale.h`: `nests`, `overlaps`, `eventFacts`, `selectScale`; `test_scale`), the definition `cortical_event_pairs` (the per-pair half: `for:` over the pair set, the existing steps unchanged), the verb `scaleselection compute`
(`pipelines/src/scaleselection.cpp`), the measures 1031 … 1040 (renumbered at the merge with the tile-pair quadrature, which took 1029-1030), `test_scale_selection` (and `test_measures`' count). The executor is unchanged; the one generic runner fix is that a labelling names the bursts of ITS OWN step (`events` input), not "the" spans output.

- **The cross-pair half is a verb, not a step.** As the spec says: a step has fixed ports, the pair set is a list. List-valued ports (through the plan, the executor's input gathering and the cache key) would let a definition hold it; not built. The verb reads stored
  rows by name: it needs `run cortical_event_pairs` first (one pass), and a rerun of the run with other options (`--replace`) changes the inputs' identity, so the selection is recomputed (the key holds each labelling's and baseline's record).
- **Each pair has its OWN global bursts.** `cortical_event_pairs` detects bursts per pair on that pair's trace (the max over its level's tiles), so the events of one episode at different pairs lie in different bursts; they are matched by their cells (nesting and overlap), not by a shared burst.
  The alternative (one burst set, declared, for every pair: WHEN once, WHERE and at which scale per pair) needs `cell_extent` to map a burst's span onto another time level's tiles; not built.
- **Selection bias of the standardized peak.** On the test the planted pair's response is 9 to 11 times the other pair's of the same episode, so the bias does not show; in noise a level with more cells has a larger expected maximum of z. `--response mean` is the named
  alternative; a response corrected for the number of cells (an expected maximum under no event) is a null-like quantity and is out of scope.
- **Events that exist at no declared pair are invisible**, and the footprint is a footprint of DETECTED events: a pair where the episode was below the extent's thresholds reads NaN, not its sub-threshold z. Reading the cell z of a pair under a cross-pair event's support
  (the nesting gives the cells) is a measurement over a partition whose members are cells of several pairs; not built.
- **Linking chains.** The cross-pair events are connected components, so A (pair 1) – B (pair 2) – C (pair 3) links A with C through B even when A and C do not overlap; with hemisphere-sized coarse events, events of unrelated places can chain through a coarse event
  (the thresholds `extent_enter_sd` / `extent_exit_sd` and `--min-link` are the caller's levers). A one-to-one assignment (each event at most one partner per pair: the largest overlap) is the alternative; not built.
- **Time tiles of a pair are dyadic from sample 0** (the pair's origin is 0): the constant-Q tiling's. A pair on another origin would pass `origin`; the verb does not read one.
- **The per-pair extents' cell z are float32-stored amplitudes** read back (the run standardized the in-memory doubles): an event's cell can differ from the verb's z in the 7th digit; only the max over labelled cells is used, so no cell changes membership.
- **No event, nothing stored.** A selection with no event in any pair stores nothing (a rerun recomputes); a pair with no event stores no `scale_event` row.
- **Cost.** The links test every event pair of different pairs (a sample-bound test first, then the cells of the two events until `min_link` pairs nest and overlap): events² bound tests, cells × cells for the overlapping ones, at most; thousands of events are fast.

## Phase 2d amendment: area weights and the tile pairs (2026-11-13)

Two changes to `cellquadrature compute`, both decided on 2026-11-13; the rest of Phase 2d stands.

**1. Area weights (a new measure definition).** The ingredients become Σ_ij a_i a_j |T_ij|² and Σ_ij a_i a_j |Im T_ij|², a the barycentric vertex areas (`nxr::ladder::vertexAreas`, the ones the tiles' own areas are sums of; every component row of a vertex weighs
its vertex's area). Why: a tile's vertices stand for unequal area, so the unweighted sum depends on how densely a region happens to be sampled (a mesh refined in one place counts that place more); the weighted double sum approximates ∬ |T(x, y)|² dA dA, which does not.
- *Trace form.* W = diag(a) on a group's rows, G = Bᵀ W B (m × m, real, once per tile). ‖W^½ B X Bᵀ W^½‖² = tr(B X Bᵀ W B Xᵀ Bᵀ W) = tr(X G Xᵀ G) for real X; so with C = Cr + i Cq the same two products as before, with G the weighted Gram: ‖Re T‖² = tr(Cr G Cr G), ‖Im T‖² = −tr(Cq G Cq G).
  In code: `groupedCongruenceNorms(..., rowWeights)` builds G from the rows W^½ B; nothing else changes.
- *Named option, method rule.* `weights: area` (default) | `none` (the old behaviour, every vertex counts equally). The option is in the options struct, the CLI (`--weights`), the cache key and the recipe. `none` keeps the old cache key exactly, so its rows are the old rows (byte for byte, tested); `area` adds
  `weight: area` and `vertex_area: barycentric` to the key, so rows of the other definition are another job's and refused unless `--replace`: old rows are never read as new. The stored measures' descriptions (1022, 1023) now say "area-weighted unless `weights: none`".
- *Decision recorded: the areas.* Barycentric (`vertexAreas`), as the tiles' areas and `patches` use; the eigenmodes' mass (mixed Voronoi) is nearly equal, but a second area definition inside one level's measures would be a source of confusion.

**2. Tile-pair quadrature.** For each pair (s, s') of tiles of the level joined by a mesh edge (`ladder::tileAdjacency`, listed once, s < s'): T_ss' = B_s C B_s'ᵀ, the block of the source cross-spectrum coupling the two tiles, where a wave running from one tile into the next lives (it is in neither tile's own block).
- *Trace form.* ‖B_s X B_s'ᵀ‖² (weighted) = tr(X G_s' Xᵀ G_s) for real X, so ‖Re T_ss'‖² = tr(Cr G_s' Cr G_s), ‖Im T_ss'‖² = −tr(Cq G_s' Cq G_s), each = Σ_ij (X G_s')_ij (X G_s)_ji. With X G_s held for every tile once per C, a pair costs m² (an element-wise product and sum) on top of the
  tiles' own 2 S m³: the pairs are almost free beside the cells. T_s's = T_ss'ᴴ has the same norms, so a pair is listed once. T is never formed (`quadrature::pairCongruenceNorms`: a function of matrices, the pairs a list of group pairs; it also returns the tiles' own blocks, so one node serves both).
  Same per-burst rule: C summed over the burst's tiles first, then the pair ingredients; a burst with a missing tile is NaN.
- *Storage, decided.* Measurements of the recording over the same Selections as the cells (the joint cells; the bursts), as array-backed rows where **rows = the level's tile pairs** (in `SL.pairs` order, which is `tileAdjacency` of the stored mesh and tiling, deterministic) and members = time tiles (or bursts): [pairs × time tiles × 2] and
  [pairs × bursts × 2] float64, NaN as the cells. The pair list itself is stored beside as int32 [pairs × 2] (`<array>_pairs`) so a row is read as a pair without recomputing the graph. No new Selection type: a closed-form pairs Selection would need a new Selection kind for what is a row index of the array, as the spatial tiles are for the cells; revisit when a statistic must
  summarise over pairs.
- *Measures* (next free codes on main and on every open branch, 2026-11-13): `tile_pair_cross_spectrum_norm2` 1029 and `tile_pair_cross_spectrum_quadrature_norm2` 1030 (reduction sum, span levels), and the derived `tile_pair_quadrature_fraction` 1112 = √(Σ|Im T|² / Σ|T|²), a ratio of sums, derived at read.
- *Option.* `tile_pairs` (`--tile-pairs on|off`, default on): its own cache key (the pair ingredients, the tile graph), so cells and pairs are separate jobs on the same inputs.
- *What it cannot show.* The same limits as the cells: a source-space quadrature is the sensors' (the kernel's spread puts a pattern into the next tile), the fraction is a summary (not the direction or speed), noise reads a high fraction on a short window. A pair covers only tiles joined by a mesh edge: a wave crossing a tile between two non-touching ones is in neither; the pairs of a tile and its
  neighbours' neighbours are not stored. Memory: the tiles' Grams (S × m²) and two m × m products per tile at a time.

**Build notes.** Built: `quadrature::pairCongruenceNorms`, the `rowWeights` argument of `groupedCongruenceNorms`, options `weights`/`tile_pairs`, measures 1029, 1030, 1112 (and `derive` for the last), `--weights`/`--tile-pairs`. Tests: `test_quadrature` (weighted and pair forms against the dense forms; refining one region 4x leaves the weighted pair energy unchanged to
under 1 % while the unweighted one scales with the point count), `test_cellquadrature` (weights area vs the dense weighted form; the pairs against the dense cross block, cells and bursts, fixed and free kernel; a wave travelling along z reads higher between touching tiles than within them, a standing pattern reads none in either; per-burst is the summed C; a rerun is a hit; `weights: none` reproduces the old values exactly), `test_measures` (the registry count).
Issues: the within-tile fraction of a small tile on a smooth wave is low (little phase change inside it), so a coarse reading of "travelling" needs the pairs; the pair list grows with the tile graph (about 3 S at fine levels), and the stored array [pairs × time tiles × 2] about 3 times the cells'.
