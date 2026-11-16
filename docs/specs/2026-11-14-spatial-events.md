# Spatial events: localization measured as bursts measure duration (spec, 2026-11-14)

## Why

Cortical rhythms are local, and the faster the rhythm the more local it is (the authors' experience of cortical recordings; Buzsáki & Draguhn 2004; von Stein &
Sarnthein 2000). Each band has a home: alpha is strongest in posterior cortex, beta in sensorimotor cortex. Travelling waves come in two classes: within
one area (a few cm, slow) and across the cortex (fast) (Muller et al. 2018). In human intracranial recordings, theta and alpha waves travel within local
clusters, each with its own frequency (Zhang et al. 2018). MEG rhythms organise into short-lived, frequency-specific, spatially restricted networks
(Vidaurre et al. 2018), and beta events are local and brief (Shin et al. 2017).

nxr-neuro measures **when** a pattern happens as an event with a duration (global bursts, family A). It has no matching way to measure **where** a pattern
happens as an event with an extent. Its pattern measures are taken over the whole cortex, and that hides local patterns:
- **A cortex-wide decomposition cannot show locality.** On the fixture (2026-11-14, joint eigenmode × frequency plane, levels 4-8), the share of energy
  at the finest scale rises with frequency, from 0.25 below 20 Hz to 0.55 at 45 Hz. But the empty room's share is flat (≈ 1.05), and the energy above
  the empty room sits at the same coarse scale (mean level ≈ 5.1) at every frequency. The rise is the noise floor taking over as brain power falls, not
  locality. A local patch spreads over all cortex-wide modes, and the instrument's blur sets their shape.
- **A cortex-wide pattern measure cancels local patterns.** The plane asks whether the whole cortex shares one lag direction. Beta looked weak there
  (R 1.2-1.5 in its own band) and clear at local rotation centres in right sensorimotor cortex (R 2-3). Two consistent local patterns in different
  regions partly cancel in a cortex-wide measure.

So localization is measured as events with a place and an extent, as bursts are measured with an onset and a duration.

## The rule

**When stays global; where comes second.** Bursts are detected on the global envelope (the standing rule: never per vertex or per channel). Inside each
burst, spatial events are cut on the cortex. A spatial event is a time event's counterpart:

| | time (exists) | space (this spec) |
|---|---|---|
| detection | global envelope above a robust threshold | inside a burst, band power above each vertex's own baseline, cut into connected patches on the glued cortex |
| extent | duration in cycles | area, and the characteristic size (the hat level where the scale-normalised response peaks) |
| dimensionless size | cycles | spatial cycles: the patch's diameter over its local wavelength (≥ 1 can hold a travelling wave; ≪ 1 is a synchronous blob) |
| storage | a spans Selection; rise-fall measurements | a Selection of patches; place, area, size, peak as measurements |
| summary | rate, occupancy | a band's home map: how often each vertex lies in a patch (a statistic) |

## The graph (definition `spatial_events`)

```
inputs:     recording (named), kernel (named), bursts (a stored spans Selection: global_bursts' kept bursts or a stage), baseline (a stored per-vertex
            statistic, below), bad (optional)
parameters: band, levels (the hat levels for scale selection, e.g. 4-8), shape morlet, spatial_shape mexhat, threshold (ratio to baseline), min_area
chunks:
  analytic       chunk   ℝ sensors → ℂ                           Morlet (default) or Meyer
  apply_kernel   chunk   ℂ → ℂ [V × levels]                      kernel → spatial hat at each level, on every glued vertex (bridge included)
  power          chunk   ℂ → ℝ                                   |z|², pointwise
  relative       chunk   ℝ → ℝ                                   ÷ the baseline (a bound constant: pointwise, not a reduction)
  frames         chunk   the field sampled on the frame tiling, inside the bursts only
  patches        chunk   frames → patches                        nxr::cores::corePatches on the relative power (the same function as vortex cores)
  scale          chunk   patches × levels → characteristic level  the level where the scale-normalised hat response at the patch's centre peaks
outputs:    spatial events (Selection + attribute measurements), home map (statistic over vertices)
derived at read: spatial cycles = diameter / local wavelength (wavelength from the stored phase gradient over the patch); relative peak; patch rate
```

- **The baseline** is the band power each vertex has outside the bursts: a measurement over a time Selection (the run minus the bursts, minus bad
  segments), stored as a statistic and named as an input. Dividing by it removes the depth and sensitivity bias of MEG source power (deep and radial
  cortex looks weak in raw maps). It is the spatial counterpart of the robust-median threshold in time.
- **The patch finder is reused.** `corePatches` does not know its input: on the curl it gives vortex cores, on relative power it gives spatial events.
  Both are stored the same way (decision 1 below).
- **Scale selection** follows Lindeberg (1998): at a patch's centre, the hat response at each level, normalised so a blob of matched size gives the same
  peak at every level; the level of the largest response is the patch's characteristic size. Stored per patch, in cm (the level's band centre).

## Pieces (each a PR, each with a unit test; goldens on synthetic data)

1. [x] **Baseline statistic** (PR #154, 2026-11-14; `run spatial_baseline`): band power per vertex over a time Selection (`measure` over the complement of the bursts), stored by name.
2. [x] **`relative` step** (PR #154): pointwise division by a bound per-vertex constant. General algebra (a diagonal scaling), its own function.
3. [x] **Multi-level `apply_kernel`** (PR #153; `levels`, `glued`, `at_level`): one operator per hat level, bound at preparation, output [V × levels]. Golden: nxr `hat_operator` at each level.
4. [x] **`scale` step** (PR #155): scale-normalised hat responses at the patch centres, the argmax level. Golden: planted Gaussian blobs of known widths on a sphere; the
   selected size tracks the planted width.
5. [x] **Spatial events stored** (`run spatial_events`, step `patches`; threshold `percentile` 99 (default, from `run spatial_threshold`; 2026-11-14) or `ratio`) as the cores are (decision 1), with attributes: burst, frame, centre vertex, centre (x y z), area, characteristic level,
   size (cm), relative peak.
6. [x] **Home map** (statistic `spatial_events.home.<recording>`, counts; fraction at read) per vertex, the fraction of burst frames it lies in a patch (a measurement over the frame partition, ratio derived at read: frames in
   a patch / frames).
7. **Regional pattern measures:** lag direction, gradient against rotation, consistency and rotation centres, measured over a spatial Selection (a
   patch, a region of the home map, or a surface-ladder tile of the band's characteristic size) instead of the whole cortex. These are the existing
   measurement functions given a spatial partition; no new reduction.
8. **One pattern or several:** for a burst with several patches, the existing `cortical_event_pairs` step measures whether they are phase-locked.

## Statistics and nulls

- **Size against frequency** (the maintainers' trend: faster is more local): the distribution of characteristic size per band, over events, as a statistic.
  Compared with the lag-free null (piece 10 of the vortex-cores spec), which keeps the data's spatial covariance: an extent beyond the null's is beyond
  what mixing alone produces.
- **The instrument's blur is a separate measurement,** never built into a tool: the same chain run on simulated point sources through the forward model
  (`headmodel_surf_os_meg` and the kernel), stored as a reference statistic that a reader may compare with. The empty room is not a reference through a
  kernel whose noise covariance came from it.
- **Home maps across runs:** the correlation of two runs' home maps (the run-repeat statistic's pattern), with BH over regions where regional.

## Regions

- **Data-driven first:** a band's regions of interest are Selections cut from its home map (connected regions above a stated occupancy).
- **Atlas labels describe them,** never define them.
- **A population home map** (e.g. alpha posterior, beta sensorimotor from a group) may be named as an optional input to restrict an analysis; it is
  never a default.

## Decisions (the maintainers delegated them, 2026-11-14)

1. **Storage of cores and spatial events** (shared with the vortex-cores port, piece 5): one member per event, a **spans Selection whose member is the
   event's frame** (several members may share a frame), with its place (centre vertex, x y z), sense or band, area, size and peak as **measurements over
   that Selection**. No new point type. Tracks take the same form (a span from first to last frame) plus the labelling event → track.
2. **The threshold:** ~~`threshold: ratio` (default, 2× the vertex's baseline)~~ **corrected 2026-11-14 (maintainers, delegated):** the threshold is set by its FALSE-ALARM RATE, measured, as the time burst detector's threshold comes from the envelope's
   own distribution. A vertex's single-frame relative power is about exponential, so 2× the baseline is exceeded by about 13.5 % of the vertices of every frame on noise alone. The default is `threshold: percentile`, `percentile` 99: the 99th
   percentile of the relative power **outside the bursts** (and bad segments), pooled over the glued vertices, read from a stored distribution (`run spatial_threshold`, a statistic `spatial_threshold.histogram.<recording>`, 4096 log
   bins, range fixed from the run's first 3000 good frames: decision 5) by name (`threshold_statistic`) and derived in preparation; refused when made from another band, levels, reference level, shape, tiling or baseline. The derived value
   is in the provenance. `threshold: ratio` (2.0) stays as the named option.
3. **The time unit is the frame;** per-burst values are measurements over a burst's frames.
4. **Scale normalisation:** each level's response is the correlation with the level's unit-energy atom (the hat operator's column divided by its L2 norm
   at that vertex), so a blob gives its largest response at the level matching its width. The size is the level's centre wavelength 2π/√λ_c in cm; the
   patch's diameter is the equal-area diameter 2√(area/π).
5. **The vortex cores' histogram range** (the cores step's open question): the first 3 000 frames, nxr's value, counted from the run's first burst
   frame, so it does not depend on the chunk size.

## Order

After the vortex-cores chain (pieces 5-8, 10): pieces 1-6 here, then the fixture check (alpha posterior, beta sensorimotor home maps; size against
frequency against the null and the point-source reference), then piece 7 and 8.

## Checklist (the architecture policy)

- **Compute functions callable with a matrix from anywhere:** relative scaling, patches, scale selection and the multi-level operator are pure functions.
- **Another input by naming it:** the recording, kernel, bursts (or a stage), baseline and any population map are named by the caller.
- **Loops and file access in io:** the chunk service delivers blocks; io stores Selections, measurements and statistics.
- **The caller chooses the data:** bands, levels, thresholds, bursts and regions are parameters or inputs.

## References

Buzsáki G, Draguhn A (2004) Neuronal oscillations in cortical networks. Science 304:1926-1929.
von Stein A, Sarnthein J (2000) Different frequencies for different scales of cortical integration. Int J Psychophysiol 38:301-313.
Muller L, Chavane F, Reynolds J, Sejnowski TJ (2018) Cortical travelling waves: mechanisms and computational principles. Nat Rev Neurosci 19:255-268.
Zhang H, Watrous AJ, Patel A, Jacobs J (2018) Theta and alpha oscillations are traveling waves in the human neocortex. Neuron 98:1269-1281.
Vidaurre D et al. (2018) Spontaneous cortical activity transiently organises into frequency specific phase-coupling networks. Nat Commun 9:2987.
Shin H, Law R, Tsutsui S, Moore CI, Jones SR (2017) The rate of transient beta frequency events predicts behavior across tasks and species. eLife 6:e29086.
Lindeberg T (1998) Feature detection with automatic scale selection. Int J Comput Vis 30:79-116.
