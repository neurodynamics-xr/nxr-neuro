# Vortex cores, their tracks and their nulls: the project's robust population methods in nxr-neuro (spec, 2026-11-14)

## Why

Earlier population work with the nxr Python implementation (`nxr.detect.vortex`, `vortex_cores`) distinguished measurements from descriptions. Two kinds of result are worth keeping:
- **Within a recording, vortex cores travel consistently through each tile, beyond a lag-free null, and the direction repeats run to run.**
- **Across people, a direction map is read only after each person's own null is subtracted.**

The same work showed what is NOT a measurement and must not be ported:
- core speed, lifetime, reversal and recurrence (they equal the lag-free null);
- phase singularities and winding counts (the travel round trip on the fixture: the net winding is right 0-25 % of the time);
- the sense of rotation from the current (chance; from the CURL MAP it is right in 88-100 % of 25-40 mm patches).

nxr-neuro has none of the pieces of the robust chain in a pipeline. This spec ports them inside the paramount architecture: pointwise nodes, every
reduction a measurement over a partition Selection, ratios derived at read, and statistics computed from Selections.

## Decisions

1. **Defaults:**
   - Temporal: a **Morlet**, centred on the band, the band's width as its full width at half maximum (the project's fixed-band rule, nxr
     `vortex.morlet_gain`).
   - Spatial: a **Mexican hat** g(λ) = λs²·e^(−λs²/2) / max, peaked (λ = 2/s²) at the centre of the level's Weyl band (nxr
     `vortex_cores.hat_operator`).
   - Meyer stays as the named option `shape: meyer` (temporal) and `spatial_shape: meyer` (spatial).
   - The rule "a spatial level IS its spatial band" becomes "a spatial level is the hat peaked at its band's centre". Hats overlap and do not add up,
     so any analysis that splits power across levels or bands (scale spectra, the joint eigenmode × frequency plane, power shares, scale selection)
     sets `meyer` explicitly. The definitions that do so are listed in piece 2.
2. **Tracks are stored as a Selection;** every track measure and statistic is built from it.
3. **The null is the project's envelope-matched lag-free surrogate**, made efficient as follows:
   - **Second-moment measures** (anything linear in a burst's covariance Q: the phase current, the quadrature fraction, the lagged share, the plane's
     consistency): **Wishart draws** of Q, 1 000 per run.
   - **Cores and tracks** (thresholds, patches, links: not quadratic): **surrogate recordings, with early stopping** (Besag-Clifford sequential
     p-values). Draws run in batches of 20, at least 20 per run and at most 100, stopping when no tile is undecided.
4. **Bursts: `global_bursts`, with the project's improvements as options:**
   - durations in cycles;
   - the rise-fall fit gate (r² ≥ 0.7, τ not at its bound);
   - onset / within / offset stages as Selections.

   Both detectors are then compared once on the fixture and on a stationary surrogate (how often each fires on noise).

## The graph (definition `vortex_cores`)

```
inputs:     recording (named), kernel (named), tiles (the surface ladder), bursts (a stored spans Selection: global_bursts' kept bursts or a stage),
            null (optional: the surrogate's ingredients, piece 10), bad (optional)
parameters: band, space_level (6), frames_per_cycle (36), core_percentile (98), link_mm (15), shape morlet, spatial_shape mexhat, seed (draws)
chunks:
  analytic       chunk        ℝ sensors → ℂ                     Morlet (default) or Meyer
  apply_kernel   chunk        ℂ → ℂ [V]                         ONE operator bound at preparation: kernel → tangent part → curl → spatial hat, evaluated
                                                                on every GLUED vertex (bridge included;: one cortex)
  frames         chunk        ℂ [V] → ℂ [V] at the frame tiling  the field sampled on the frame partition (piece 4); only frames inside the bursts
  cores          chunk        frames → cores                    |Re w| ≥ the threshold statistic; connected patches per sense per frame (piece 5)
  tracks         accumulate   cores → tracks                    link frame to frame, same sense, ≤ link_mm; open tracks carried across chunks (piece 6)
after:
  crossings      after        tracks × tiles → per tile × sense: Σ unit crossing vector (3, tangent at the tile centre) and the count (piece 7)
outputs:    cores (Selection + attribute measurements), tracks (Selection + attribute measurements), tile_crossings (measurement over tiles × stages)
derived at read: R = |S| / n; the unbiased ρ̂² = (|S|² − n) / (n(n − 1)); the data-to-null crossing ratio
statistics: run_repeat (piece 8); null p-values (piece 10)
```

The threshold (piece 5) needs the field's distribution over the burst frames before cores are cut. That is the runner's two passes (single-events
spec, decision 1): the first pass computes the threshold statistic, the second cuts cores.

## Pieces (each a PR, each with a unit test; each port with a golden against nxr)

1. **Morlet in `analytic`, `band_filter` and `envelope`** (`shape: morlet`, the new default; `meyer` named).
   - The gain is nxr's `morlet_gain(f, centre, fwhm)` on the taps route; the centre and FWHM come from the band.
   - Golden: nxr `morlet_gain` and a band's analytic signal on a synthetic series.
2. **The Mexican hat as `apply_kernel`'s spatial shape** (`spatial_shape: mexhat`, the new default; `meyer` named).
   - Eigenmode first: the gain on the stored modes. The modes must reach the point where the hat's gain falls below 1e-3; otherwise Chebyshev
     second, announced.
   - Definitions that keep Meyer explicitly: `cortical_cell_pairs`, `cortical_event_pairs` (scale selection across levels), and any plane or scale
     spectrum.
   - Golden: nxr `vortex_cores.hat_operator` on a sphere.
3. **The curl of the tangent current, complex.**
   - `apply_kernel` takes a `form` (`curl_tan`, `div_tan`, `normal`) folded into the operator at preparation, so the chain stays one linear map
     before any non-linear step (D106).
   - Golden: nxr `Cortex.readings` (curl) on a sphere with a planted rotating current.
4. **The frame tiling:** a closed-form time partition, `frames_per_cycle` samples of the band's centre (36 by default; the project's 10° step). Stored as
   a Selection like the chunk grid.
5. **Cores (family B).**
   - **Threshold:** a percentile statistic of |hatted curl| over the burst frames. It uses `nxr::histogram`, nxr's `_core_threshold` (4 096 log
     bins), and is stored as a statistic.
   - **Cores:** connected above-threshold patches per frame and sense on the glued mesh (nxr `_core_patches`).
   - **Storage:** a Selection `vortex_cores.cores.<recording>`, one member per core (its span the core's frame: point events, `point_events.h`). Its attributes are measurements over that Selection: frame, sense, centre
     vertex, centre (x y z), area, peak.
    - **Built (2026-11-14):** `run vortex_core_threshold` (the histogram, statistic `vortex_core_threshold.histogram.<recording>`) and `run vortex_cores` (reads it by name); README, *Vortex cores*.
6. **Tracks (family C).**
   - **Linking:** greedy nearest neighbour, same sense, ≤ `link_mm`, consecutive frames only (nxr `_track_stats`). It is an accumulator carried
     across chunks.
   - **Storage:** a Selection `tracks.<recording>` (one member per track) plus the labelling core → track. Track attributes are measurements: first
     and last frame, n_frames, path_length, displacement.
   - These are descriptions, not evidence: speed and lifetime equal the null.
7. **Crossings per tile.**
   - A crossing is one track's steps inside one tile, carried to the tile centre (the vector heat method's transport, `ladder::lcResultant`;
     nxr `vortex_cores.Tiles.crossings`), summed and made a unit vector.
   - Measurement over tiles × sense × stage: [rows × members × 4], the vector then the count (the vector-measurement convention of #139).
   - A crossing inside the callosal bridge counts in a `bridge` member.
   - Derived at read: `crossing_consistency` R and the unbiased `crossing_rho2`.
8. **Run-to-run repeat:** a statistic over two runs' tile_crossings.
   - **Per tile:** the dot product of the two runs' mean resultants, for tiles with ≥ 10 crossings in both.
   - **Whole recording:** the count-weighted mean cosine.
   - Its p comes from the null (piece 10), with Benjamini-Hochberg over the tiles.
9. **`global_bursts` options (decision 4):**
   - **`durations: cycles`:** at least 3 cycles, merged within 1 cycle.
   - **`fit_gate`:** keep a burst only if r² ≥ 0.7 and τ is not at its bound. The result is a Selection of the kept bursts.
   - **`stages`:** onset (t_on ± 2τ_rise), within (the plateau) and offset (t_off ± 2τ_fall) Selections, from the stored rise-fall measurements.
   - **Dev check:** both detectors on the fixture and on a stationary lag-free surrogate.
10. **The null (decision 3).**
    - **The surrogate as a graph node, `lagfree_surrogate`:** seeded white noise generated per chunk from the sample index (so the result does not
      depend on the chunk size), then the band's spectrum as taps, the mixing √(Re cov), and the ratio to the data's one-cycle RMS envelope.
      - Its ingredients are statistics stored beforehand: the run's band cross-spectrum (`crossspectrum compute`) and the data's envelope series.
      - A draw is the same graph run with `seed: i`; the seed is in the cache key.
      - Only the per-draw outputs are stored (tile crossings, small). The surrogate itself is never written, which corrects step 6 of the
        single-events spec: 100 stored surrogate recordings would be ~40 GB per run.
      - **Early stopping:** draws in batches of 20, at least 20 and at most 100, stopping when every tile's sequential p is decided (Besag-Clifford,
        h = 10 exceedances).
    - **Wishart draws for second-moment measures:** a statistic node. Per burst, Q_null ~ complex Wishart(Re Q_b / ν, ν), with ν the burst's
      effective samples (its duration × the band's effective bandwidth, nxr `vortex.dof_fraction`); 1 000 draws.
      - Applies to the burst phase current (#139), the quadrature fraction, the lagged share and the plane's consistency.
      - It complements the sign-flip test: sign flips ask whether a direction is shared across bursts; the Wishart null asks whether lags exist at all.
    - **The group subtraction:** each subject's null mean resultant per tile, stored, so a group map uses data minus own null (the project's `MODE=excess`).

## Not ported (the population work found them null or withdrawn)

- Core speed, lifetime, reversal and recurrence as evidence (they are stored as descriptions only).
- Phase singularities and winding counts.
- The rotation sense from the current, the naive group direction map, and single-cycle examples as evidence.

## Traps carried (the project's, 2026-10-30)

- The lag-free null is envelope-matched and has the data's own spectrum, never flat.
- The unbiased ρ̂² is used whenever places with different crossing counts are compared. Crossing counts are stored, and the data-to-null crossing ratio
  is reported.
- Each subject's null is subtracted before any group direction map.
- Beta needs a check for alpha's harmonic; gamma needs a check for muscle (bad segments excluded).
- An empty room passed through a kernel whose noise covariance came from that empty room is not a reference.
- Speeds are reported only as regional medians over ≥ 40 mm.

## Later (not in this spec)

- **The group layer:** each subject's tile vectors carried to the template (fsaverage) through its registration, the group G = |Σ m_s| / Σ |m_s| per
  vertex, and its null (each subject's map turned by a random angle about the normal, BH over vertices). This needs a resampling step nxr-neuro does
  not have.
- **The forward-model test of rotation against lagged generators** (head model, MNE, this same chain on planted sources).

## Checklist (the architecture policy)

- **New compute functions callable with a matrix from anywhere:** the Morlet gain, the hat gain, the composed operator, patches per frame, linking,
  crossings, the surrogate generator and the Wishart draws are pure functions of arrays.
- **Another input by naming it:** the recording, kernel, burst Selection (or a stage) and seed are named by the caller.
- **Loops and file access in io:** the chunk service delivers blocks; tracks are an accumulator; io stores Selections, measurements and statistics.
- **The caller chooses the data:** bands, levels, bursts, stages and draws are parameters or inputs.
