# The phase current of the cells and of the bursts: where a burst's phase travels (spec, 2026-11-13)

## Why

On the fixture (dev data `2026-11-13-common-patterns`), the method that found the common spatiotemporal pattern of alpha bursts was the PHASE CURRENT of
the band field at a spatial level, summed per spatial tile over each burst. Its ratio to the tile's power is the power-weighted phase gradient, whose
negative is the direction the phase travels.
- **The result:** one propagation pattern recurs in 80-95% of the bursts of two runs, matches across the runs (|cos| 0.95), and is absent in the empty room.
- **The contrast:** amplitude did not propagate (0 ms lags).
- **The missing piece:** nxr-neuro has the node (`phase_current`, Σ_c Im(z̄_c ∇z_c), orientation-free on a free kernel's x y z components) but cannot
  measure it over the cells or roll it up over stored bursts.

This spec adds that, inside the architecture: measurements over the joint partition, a roll-up over a stored event set, and a ratio derived at read.

## The rules followed

- **Reductions are measurements over partitions.** The phase current and the power are reduced over the joint cells (spatial tile × time tile), then over
  each burst's cells (the exact roll-up of the cells it covers).
- **The ratio is derived at read.** `phase_gradient` (rad/m, three components) = mean `phase_current` / mean `power`, as the lookup derives `sd`.
  A burst's gradient is the ratio of its rolled-up sums, never a mean of cell ratios.
- **No orientation assumption.** The current's x y z components are each a smooth band-limited field, so Σ_c Im(z̄_c ∇z_c) needs no normal and no sign
  convention. A fold's orientation flip does not enter.
- **The level IS the band.** The joint partition's spatial level is the spatial band's level, as for amplitude.

## The graph (definition `cell_phase_current`)

```
inputs:     recording (each, named by the caller), kernel (named), tiles (the surface ladder), cells (the joint partition of the pair),
            bursts (a stored spans Selection, e.g. global_bursts.bursts.<recording>; optional), bad (optional)
parameters: band [8, 16], space_level 6, method, transform auto (eigenmode first), modes, order, tolerance, bad_segments
chunks:
  analytic       chunk        ℝ sensors → ℂ                           the band (unchanged)
  apply_kernel   chunk        ℂ → ℂ [3V]                              K with the level's Weyl band folded in (unchanged; eigenmode first)
  phase_current  chunk        ℂ [3V] → ℝ³ per vertex                  Σ_c Im(z̄_c ∇z_c) (unchanged node)
  power          chunk        ℂ [3V] → ℝ per vertex                   Σ_c |z_c|² (unchanged node)
  measure        accumulate   phase_current over cells → per cell, 3 components: the area-weighted mean        NEW measure over a joint cell
  measure        accumulate   power over cells → per cell: the area-weighted mean                                NEW measure over a joint cell
after:
  rollup_events  after        each measurement over the stored bursts → per (spatial tile × burst): the count-weighted mean over the burst's cells   NEW: stored events, joint input
outputs:    cell_phase_current, cell_power (measurements over the joint cells); burst_phase_current, burst_power (measurements over the bursts' Selection)
derived at read: phase_gradient = phase_current / power (per component, rad/m); propagation = −phase_gradient; speed = 2πf / |phase_gradient|
```

## Pieces to build

1. **`measure` over a joint cell: `power` and `phase_current`.**
   - `power` of the `power` step: the area-weighted mean per cell, as `amplitude` is of `magnitude`.
   - `phase_current` of the `phase_current` step: a vector3 input (3 rows a vertex). Per cell the area-weighted mean of each component.
   - The plan checks the input's maker (as it does for `magnitude`) and refuses any other pairing with its reason.
2. **A vector measurement stored.** The array is [rows × members × 4]: components 0-2 the vector, 3 the count. That gives three `selection_measurement`
   rows, `component` 0, 1, 2, plus the count row (component 3), the same convention as the existing [rows × members × 2] with the count last.
   - Readers that assume component 1 is the count are checked and fixed. The count stays the measure `count`.
   - The registry gets `phase_current` (nxr-neuro range: vector, mean-reducible by count) and the derived `phase_gradient`.
   - `power` is the Studio's code 11.
3. **`rollup_events` over STORED events and a joint measurement.**
   - `events` may be a stored spans Selection (an input `selection: spans`), not only this run's `detect_spans` output.
   - A global event (no row) holds on every row.
   - x may be a measurement over a joint partition: the rows are the spatial tiles, the members its time tiles.
   - The roll-up is the count-weighted mean of the cells over the burst's time tiles.
   - The measures: amplitude, envelope_max (unchanged), and now power and phase_current.
   - The span set's time line must be the joint partition's recording, and its spans must snap to the time level's tiles (as `detect_spans`' do).
     Otherwise the run is refused, naming the span.
4. **The derived measure `phase_gradient`** in `io/src/measures.cpp` (`derived_measurement_values`): component-wise phase_current / power.
5. **The definition `cell_phase_current`**, `docs/reference.md` section, and the CLI (no new verb: `run cell_phase_current`).

## Tests

All on synthetic data.
- **A planted travelling wave:** a sphere or lattice cortex, a source plane wave at a known direction and speed in the band, a free kernel (identity on
  three components).
  - The cells' and the bursts' phase_gradient point along −k within a tolerance, with |phase_gradient| = |k|.
  - A standing wave gives |phase_gradient| ≈ 0.
- **Orientation-free:** flipping the sign of the current on half the vertices' components does not change the result.
- **The roll-up is exact:** a burst's value equals the count-weighted mean of its cells, computed directly. It does not depend on the chunk width or the
  sub-block width.
- **Refusals:**
  - `phase_current` over cells of a field that is not the `phase_current` step's;
  - a joint partition whose level is not the band's;
  - stored spans on another recording;
  - spans that do not snap to the time level's tiles.
- The count row shared with the other measures of the cells (the count-row rule).

## Later, for the maintainers: the statistics across the bursts (not built here)

The dev result used three statistics across the bursts. They are statistic nodes (field → measurement → statistic), and each uses a null. A null is a
choice for the maintainers, so none is built until the maintainers decide:
- **Direction consistency:** |Σ_b F_b| against random sign flips of whole bursts. The null: with no consistent direction, each burst's field is as
  likely reversed.
- **The common pattern:** the leading singular vector of the bursts' tile fields and its share of their variance, and each burst's score on it (+ forward,
  − reversed).
- **Replication:** one run's pattern scored on another run's bursts (cross-run prediction; no null needed beyond the sign).

## Checklist (the architecture policy)

- **New compute functions callable with a matrix from anywhere:** the per-cell area-weighted sums of a vector field and the roll-up are pure functions of
  arrays.
- **Another input by naming it:** the recording, the kernel and the stored bursts are named by the caller.
- **Loops and file access in io:** the chunk service reads; io stores the rows and arrays.
- **The caller chooses the data:** the band, the level and the burst set are parameters or inputs; nothing is hard-wired.

## As built (PR #139, 2026-11-13): deviations from the text above

- **Measure names:** a vector measure is three measures, `phase_current_x/y/z` (codes 1041-1043) and the derived `phase_gradient_x/y/z` (1113-1115), not
  one name with components 0-2. The unique key of `selection_measurement` holds the measure but not the component, so three rows of one name would
  collide. The array is still [rows × members × 4], the count last.
- **The burst set** is found by the parameter `bursts_pipeline` (default `global_bursts`) and the recording named in the spans' provenance. A `where` on a
  name cannot name one burst set per recording in an `each` run. The input is required.
- **Gradient length:** the derived gradient is the gradient of the BAND FIELD at the level, not |k| of the planted wave. On the synthetic test it is 0.43
  of k's tangent projection; the direction is exact (cosine 0.99). A band-pass on a curved, coarse mesh shortens it.
- **Speed (correction of the text above):** leakage and the band shorten the phase gradient, so a speed 2πf / |∇φ| is OVERESTIMATED, not shrunk.
