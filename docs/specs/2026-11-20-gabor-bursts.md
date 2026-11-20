# Gabor bursts: burst detection in time by matched Gabor atoms, with an optional spatial adjustment (spec, 2026-11-20)

Status: **design, for the maintainers' review before code.** A burst-detection pipeline beside `band_bursts` and `global_bursts`. It is ported
from the lab's staging detector, which runs on sensor-level MEG, and decomposed into nxr-neuro nodes.

## 1. The method

A burst is matched by a **unit-energy Gabor atom** with a FREE width:

ψ(t; t₀, f, σ) = (πσ²)^(−1/4) e^(−(t−t₀)²/2σ²) e^(i2πf(t−t₀))

The coefficient W(t₀, f, σ) = ∫ x ψ* dt is a matched filter. For a Gaussian burst A e^(−t²/2s²) cos(2πft + φ), |W| is largest at σ = s, where
|W| = (A/2) π^(1/4) √σ. So one peak of |W| over (t₀, f, σ) gives the burst's centre, frequency, width (FWHM 2.355σ), cycles, amplitude and phase.
Unlike Morlet, the width is not tied to the frequency: a short burst and a long one at the same frequency are told apart by σ.

The pipeline is **temporal first**: it finds bursts per channel in time. A **spatial adjustment** follows only when the channels have known positions
(sensor locations). It moves each burst's peak sensor to the extrema of the burst's signed map: its source and sink.

## 2. The pieces, as nodes

| # | piece (staging) | nxr-neuro node | kind | new or existing |
|---|---|---|---|---|
| 1 | the atoms on a grid: widths `per_oct` an octave, frequency steps σ_f/2, each width at its own decimated rate | **`signal.gabor`**, on the time filter bank: a new `Family::Gabor` (section 3) | compute, chunk | new family + node |
| 2 | each atom's own level: median and robust spread (1.4826·MAD) of \|W\| over time, per channel and frequency row | `measurement.summarize` (median, mad) over the time partition | reduction (a measurement) | existing |
| 3 | candidates: local maxima of \|W\| in (f, t) (3×3) above median + k·spread | **`select.events.peaks`**: local maxima of a field on a product grid, above a level read from a statistic | selection maker | new rule |
| 4 | refine: a parabola through log\|W\| in f (±σ_f/2) and log σ (×2^(±1/8)), twice, by the direct inner product; then amplitude, phase, cycles | **`select.gabor_fit`**: per candidate, on the samples around it (like `select.risefall`) → f, σ, amplitude, phase, cycles, \|W\| | per-event measurements | new |
| 5 | the width test: \|W\| peaks along σ too, inside the ladder; at least `mincyc` cycles; f inside the band | measures of `select.gabor_fit` (`width_peak`, `cycles`) → `select.events.spans` `keep_where` | selection | existing rule |
| 6 | merge across widths: the strongest first; drop a peak within a kept one's time and frequency resolution | **`select.events.suppress`**: non-maximum suppression of point events within each event's own resolution | selection maker | new rule |
| 7 | the fit test: R² of the atom against the band's signal in its half-max window | measure `r2` of `select.gabor_fit` → `keep_where r2 ≥ r2min` (0.7 by default) | selection | existing rule |
| 8 | edges: the tangent at the steepest rise and fall, read on an envelope at the burst's f with σ_r = max(σ/2, 3/(2πf)), meeting the trough's level (default); or the fit's half max (centre ∓ 1.18σ) (option) | the envelope: `signal.analytic` (Morlet, the burst's f and width); **`select.events.spans` rule `edges`** (`tangent` · `half_max`) | selection | new rule |
| 9 | across channels: a global window where at least one channel bursts; the channels' bursts kept inside it | `select.events.spans` (`union` of the channels' spans, by time–frequency overlap) | selection | existing rule (check `union` covers time–frequency overlap) |

### The optional spatial adjustment (when the channels have positions)

| # | piece | nxr-neuro node | new or existing |
|---|---|---|---|
| 10 | the sensors' graph: neighbours on the helmet | **`geometry.points_graph`**: a graph on a points manifold from its positions (Delaunay of the surface they lie on, or k nearest); a `graph` manifold | new |
| 11 | the burst's map: its atom (t₀, f, σ) read on every sensor, complex | **`select.gabor_map`**: per event, the atom's coefficient on every channel → a field on sensors × events | new |
| 12 | the signed map Re(w e^(−iφ_c)), the phase taken at the peak sensor c | `signal.field_form` (a new form `signed_at`: the real part after removing a reference phase) | new form |
| 13 | climb to the extrema: from the peak sensor up to the local maximum (source), from the antiphase sensor down to the local minimum (sink), across graph neighbours | **`geometry.climb_extremum`**: a field on a graph manifold; from a start vertex, to its local maximum or minimum | new, general (any field on a mesh or graph) |
| 14 | refine between sensors: a quadratic over the extremum and its neighbours in its tangent frame | **`geometry.refine_extremum`**: the sub-vertex position of an extremum | new, general |
| 15 | the measurements: source and sink sensors and positions, how far each moved, the source–sink separation | measures over the events | existing measurement system |

Pieces 13 and 14 are general geometry. They serve any field on a mesh, the cortex included, not only sensors.

## 3. The Gabor family in the time filter bank

`timefilterbank.h` gains `Family::Gabor`. Its members are indexed by (σ, f), not by f alone.
- **Grid.** Widths σ_k = σ_min·2^(k/per_oct) up to σ_max. At each width, frequencies from max(f_min − σ_f, min_fσ/σ) to f_max + σ_f in steps of σ_f/2,
  where σ_f = 1/(2πσ) (a margin of one σ_f, so a peak at the band's edge is told from a maximum). Atoms with f·σ < min_fσ (about 1.2 cycles at half max)
  are skipped.
- **Gain: the matched-filter (unit-energy) normalisation**, Ψ(ν) = (πσ²)^(−1/4) √(2π) σ e^(−2π²σ²(ν−f)²). It is a new gain convention beside the bank's
  "peak G", recorded on the bank. A Gabor member is not unit-peak, because the matched filter needs unit energy.
- **Analytic** (one-sided) only, for a real signal.
- **Each width at its own rate.** Width k is computed on its band only and decimated to the coarsest time grid that keeps 3 samples a σ. The output is a
  line per width, tiled from the recording's time line (`made_from: tiled`). Memory holds one channel and one width at a time.
- **Not tight** (a matched-filter bank, not a frame).

Defaults follow the staging detector: σ from 25 ms to 640 ms, 4 widths an octave, k = 4, 2 cycles, R² ≥ 0.7, tangent edges.

## 4. The pipelines

**`gabor_bursts`** (temporal): a recording and a band (f_min, f_max).

    recording → signal.gabor (each width) → |W| → measurement.summarize (median, mad: per channel × frequency, over time)
              → select.events.peaks (above median + k·spread) → select.gabor_fit (per candidate: f, σ, A, φ, cycles, width_peak, r2)
              → keep_where (width_peak, cycles ≥ mincyc, f in band) → select.events.suppress → keep_where r2 ≥ r2min
              → edges (tangent | half_max, on the envelope at each burst's f) → the channels' bursts (spans, with their measurements)
              → union → the global windows (spans), the channels' bursts kept inside

Output: the bursts per channel, a spans selection on the time line, with per-burst measurements:
- t₀, onset, offset, f, σ, FWHM, cycles, amplitude, phase, R², the channel;
- the global windows over all channels.

**`gabor_bursts_localised`** (spatial adjustment). Its inputs are the bursts of `gabor_bursts`, the recording, and the channels' manifold with positions.

    channels (positions) → geometry.points_graph
    bursts, recording → select.gabor_map → field_form signed_at (the peak sensor's phase)
              → geometry.climb_extremum (source from the peak sensor; sink from the antiphase sensor) → geometry.refine_extremum
              → measurements per burst: source, sink, their positions, moved distance, separation

Refused when the channels have no positions, with a message naming what is missing.

## 5. What is not ported here

- **The heat-kernel blobs on the sensor graph** (grouping atoms into spatial events, the staging detector's stage B). They are replaced by the
  temporal union (piece 9) and the extrema adjustment (pieces 10–15). They can follow as their own pipeline if wanted.
- **The coexact localisation** (the joint peak of the coexact quaternion over sensors). It belongs to the analysis of a burst, after detection.

## 6. Tests (synthetic only)

- **Planted Gabor bursts** in noise, with known (t₀, f, σ, A, φ), recovered within their resolution. The amplitude follows A = 2|W|/(π^(1/4)√σ).
- **Two bursts at the same f** with different σ are told apart.
- **A planted dipole map on a synthetic sensor helmet:** the climb finds its extrema, and the refined positions fall between sensors within a tolerance.
- **Chunk invariance:** the same bursts whatever the chunk size.

## 7. Plan (PRs, each with its tests)

1. `Family::Gabor` and `signal.gabor` (piece 1, section 3).
2. `select.events.peaks`, `select.gabor_fit`, `select.events.suppress`, the `edges` rule; then the `gabor_bursts` definition (pieces 2–9).
3. `geometry.points_graph`, `select.gabor_map`, the `signed_at` form, `geometry.climb_extremum`, `geometry.refine_extremum`; then
   `gabor_bursts_localised` (pieces 10–15).

## Checklist (the architecture policy)

- **Each compute is a function of arrays:** the bank, the atom fit on a window, the climb on a field over a graph.
- **Reductions over time are measurements over partitions.** The atom's level is a summary over the time partition; it is never computed inside a node.
- **Detection and suppression are selection makers.** Per-burst values are measurements over the events.
- **The recording is read by the chunk service.** The largest width sets the halo (3σ_max ≈ 2 s).
