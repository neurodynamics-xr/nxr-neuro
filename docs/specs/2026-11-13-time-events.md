# Labelling the time axis with events: windowed band amplitude and its bursts (spec, 2026-11-13)

## Why

The maintainers defines event detection (2026-11-13) as standard time-series event labelling: the time axis of a recording labelled with events.
- **The events:** bursts as Rheome detects them, spindles, slow waves, noisy bad segments, the imported Brainstorm events, and later sleep stages.
- **The means:** the selection/measurement system, as the Studio's Selection classes do: the stored zarr, a lookup and measurement functions.

The audit (`docs/audits/2026-11-13-time-event-labelling.md`) found that nxr-neuro has every part but two. This spec adds those two, and the first
detector built from them: a band's bursts.
- **Windowed measurements of a chain's output** (the sampled layer): `docs/reference.md` lists it as not built yet ("`measure` over a stored time partition").
- **The step from windowed measurements to a labelling**: the Studio's `searchAtlas` pattern.

## The rules followed (no new storage, no new layer)

- **A windowed band amplitude is a measurement, not a Field**.
  - The measurement function `amplitude` (mean |z|; the Studio's registry code 10) or `power` (mean |z|², code 11) of the recording is taken over the
    tiles of the constant-Q level OF THAT BAND. "The level IS the band" (the Studio's D84, `measurements.ts`), which is how a measurement row, keyed
    (selection, tile, recording, measure), says which band it holds.
  - The plan refuses a band that is no level's band, and its message lists the levels' bands.
- **Stored as the envelope and `measure over time` store:**
  - an array [rows × tiles × components] beside the recording;
  - `selection_measurement` rows (selection = the level, `of_field_id` = the recording, the measure, `array_path`, `component`);
  - how they were made named by `computed_by` (the recipe policy of `measure compute`): the pipeline, its version and its cache key's hash.
- **A detected event set is a `spans` Selection on the recording's time line, in the importer's form** (`nxr::bst::time_spans`; the model's D143):
  - one row per event, [start, stop) in seconds;
  - `code` = the event type through a dictionary named after the Selection;
  - per-event channels in `params_json.channels`.
  - Its `params_json` also records how it was made, as the Studio's `searchAtlas` records its query in `params_json.search`: the pipeline, its
    parameters, the thresholds used, and the cache key.
- **The thresholds are statistics computed at detection time and never stored:**
  - per row, the median and robust SD (1.4826 × MAD) of the tile values over the run's span;
  - as Rheome's `detect.spindle` computes them, as the Studio computes them (`spindleEnvelopeDetail`), and as every audited reference does;
  - the statistics layer (decision 1 of the event spec) stays open;
  - the values used are recorded in the Selection's `params_json`.
- **A burst's peak and mean amplitude ARE measurements over the bursts Selection** (amended 2026-11-13, maintainers). The bursts are
  a partition of the time axis like the tiles: measured within each burst (by the exact roll-up of the tile measurements it covers) and stored, which the statistics
  across the bursts read. A burst's peak is the max of its tiles' maxima, so `band_bursts` also stores the tile maximum. Spec: `docs/specs/2026-11-13-burst-statistics.md`.
  (The first version of this spec said they were not stored; that was wrong.)

## The graph (definition `band_bursts`)

```
inputs:     recording (field: recording, each), ladder (selection: ladder on recording.time: the constant-Q tiling, `constantq compute`),
            bad (the recording's bad segments, optional)
parameters: band [8, 16] Hz (a level's band), combine each|mean, enter_sd 1.5, exit_sd 1.0, merge_s 0.1, min_s 0.4, max_s null
chunks: over recording
  band_filter   chunk       ℝ [C x T] → ℝ                the band (unchanged step)
  envelope      chunk       ℝ → ℝ |z|                    the band's analytic amplitude (unchanged step)
  measure       accumulate  |z| over the ladder's level of the band → amplitude per row per tile: the Studio's reduceTiles over the good samples
after:
  detect_spans  after       amplitude [rows x tiles] → spans: per row (or the mean over rows) the median and robust SD over the good tiles;
                            crossings ≥ median + enter_sd·SD extended while ≥ median + exit_sd·SD (hysteresis); spans closer than merge_s joined;
                            kept when min_s ≤ length ≤ max_s (nxr::intervals: crossings, mergeSpans, keepSpans; median, mad)
outputs:
  amplitude     measurement   the sampled layer, rows on the level (above)
  bursts        spans         the event set (above), event type "burst"
```

- **The reference parameters** are Rheome's alpha episodes (`flow.labelalpha` → `detect.spindle` in envelope mode): 1.5 / 1.0 robust SD, merge 0.1 s,
  min 0.4 s, no max, the baseline over the whole input.
  - `combine each` applies them per channel, as `detect.spindle` does; `combine mean` applies them to one trace, the mean over channels.
  - Rheome's 0.2 s moving mean is the tile itself here: a windowed analysis reduces.
  - Rheome's rise/fall fit (`detect.risefall`) is a later, separate step that describes a span.
- **Durations and gaps in seconds are compared in whole tiles:**
  - a span's length is its tiles × the tile width;
  - spans are joined when the tiles between them span ≤ merge_s;
  - the provenance states the tile width.
- **A tile is good when the span covers it whole and it holds no bad sample** (its count equals the width). Other tiles are NaN in the array, and no
  statistic or span uses them.
- **The span is the caller's** (chunk 0 by default, `--chunks all` for the whole recording). The statistics are over that span, and say so.

## What it cannot resolve

- **Time resolution is the band's tile** (alpha at 1200 Hz: 1024 samples, 0.85 s).
  - A burst's onset and offset are known to a tile.
  - The band's response can carry an edge into the next tile: on the synthetic test, a burst 10 times the background extends one tile past its true end.
  - So an edge is the tile holding it or the next one out.
- **The band is the level's**, an octave at the hertz anchor (alpha 8–16 Hz). A narrower band (8–12 Hz) is not a level and is refused.
- **These are sensor quantities.** A burst found on a channel says nothing about where in the cortex it is.

## Runner changes (pieces 1 and 2; nothing else changes)

1. **`measure`** takes an optional `over` (a constant-Q ladder) and the measures `amplitude` and `power`.
   - The plan checks D84: the input is a band's envelope (`amplitude`) or power (`power`), and some level of the ladder has that band.
   - The fold is per tile (reduceTiles on runs of good samples).
   - The output is a measurement, not a Field.
2. **Outputs of kind `measurement` and `spans`**, stored as above.
   - A rerun with the same cache key is a hit: rows found by their `computed_by`, a Selection by its `params_json.cache`.
   - Another key's outputs are refused unless `--replace`.
   - `find_cached` reads a Selection's key from `params_json` (Selections have no `producer_json`).
3. **The after step `detect_spans`**, calling the existing pure functions of `nxr::intervals`.

## Tests

`test_band_bursts`, a synthetic recording (never a real dataset):
- 3 channels at 1200 Hz, 10 Hz bursts planted at known times on a noise floor, a constant-Q ladder written;
- the amplitude rows stored on the alpha level (8–16 Hz), equal to the direct mean of |z| per tile;
- the planted bursts found to within a tile, with no others;
- the same with another chunk width;
- a bad span drops its tiles;
- a rerun is a hit;
- refusals: a band that is no level's (8–12 Hz), and no ladder.

## Checklist (the architecture policy)

- **Pure functions:** yes. The new compute is `nxr::intervals` (existing), median and mad (existing) and reduceTiles (existing).
- **Another input by naming it:** yes. The recording, the band and the parameters are the caller's.
- **Loops and file access in io:** yes. The chunk service reads, and io writes the rows and the Selection.
- **The caller chooses the data:** yes. The span, the band and the combination are parameters, and the level follows from the band by D84.
