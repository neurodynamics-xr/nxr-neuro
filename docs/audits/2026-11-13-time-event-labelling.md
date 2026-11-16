# Audit, 2026-11-13: labelling the time axis with events (read-only)

**Purpose.** Event detection in nxr-neuro is standard time-series event labelling: the time axis of a recording labelled with events.
- the events: bursts as Rheome detects them, spindles, slow waves, noisy bad segments, the imported Brainstorm events, and later sleep stages (not now);
- all served by the selection/measurement system, as the Studio's Selection classes are, through the stored zarr, a lookup and the measurement
  functions.

This audit records what the references do. It is not a design. Every claim was read in the code, and the key ones checked by hand.
- Repos and commits: the Studio `nxr-cortical-flow` (4f208763e), Rheome (`rheome/+rheome`), nxr (`nxr/src/nxr`) and nxr-neuro (main, 9cd3b03).
- No dataset content was read.

## 1. The stored form of an event labelling (the model; the Studio reads it, nxr and nxr-neuro write it)

- **A `spans` Selection on the recording's time Line** (`model.sql:486-531, 666`; the Studio's D143, `docs/superpowers/plans/2026-10-02-one-schema.md:29`):
  - a member per occurrence, as a `selection_span` row (ordinal, start, stop in seconds, code);
  - an instant has start = stop;
  - `code` = the event TYPE, read through the Selection's dictionary (a `dictionary_entry`: name, colour);
  - cell '1', unit 's';
  - `of_field_id` = the recording.
- **The same type in closed form** (`params_json.tiling = {origin, width, hop, count}`, no rows) is a tiling, not an event set: the chunk grid, the
  envelope windows, the constant-Q levels.
- **The Studio's one Selection class** (`backend/src/entities/selection.ts:63`) gives a time `spans` selection:
  - `at(k)`: the member as a `span`, with its code and dictionary;
  - `count`;
  - `indexAt(t)`: O(1) in closed form, a scan of the rows otherwise; members are half-open [start, stop).
  - There is no separate class per type.
  - In the Studio UI a `spans` port (`studio/src/studio/ports.ts:247`) is drawn on the Timeline as clickable bars (`Timeline.tsx:828`).
- **nxr-neuro's importer writes exactly this form** (`io/src/bst_events.cpp`, `nxr::bst::time_spans` in `io/src/bst_entities.cpp:102`; a port of nxr's
  `convert/events.py:198`):
  - `<rec>_events` holds every Brainstorm event occurrence:
    - code = the type in a subject-scope dictionary named `<rec>_events`, with Brainstorm's colours;
    - per-occurrence channels go in `params_json.channels` {ordinal: [channel indices]}.
  - `<rec>_bad_segments` holds the occurrences whose type name contains "bad" (with stop > start), in a dictionary of its own.
  - nxr also imports MNE annotations (`BAD_*` become bad segments), BIDS `events.tsv`, and Brainstorm's own bad-segment detection results
    (`import_bad_segments`). Scored sleep stages arrive as `stage/<label>` spans. NOTHING is detected in nxr; everything is imported.
- **Consumers read `<rec>_bad_segments` as an exclusion mask:**
  - nxr: `pipelines/measurement_tables.py:178` (`good_mask`, ± 0.5 s), `pipelines/parameters.py:255`;
  - nxr-neuro: the chunk service's `bad` spans, the runner's `exclude` input.
  - A detector's output must not take that name unless it means exclusion.

**Correction to the event spec's decision 2 (revision 3):** it said "code = the span's ordinal, so a code identifies a span, never an event type". The
model and the importer use code = the event TYPE. Per-span attributes are therefore not keyed by a per-span code in any existing writer.

## 2. Measurements and the lookup (the Studio)

- **The registry** (`backend/src/db/measurements.ts:91-134`):
  - stored measures: mean, max, min, mean_square, sum, abs_max, area, n_elements, `amplitude` (10, mean |z|), `power` (11, mean |z|²), the mergeable
    sums, envelope_sum/max, gradients, tensors;
  - derived at read: variance, sd, rms, peak_to_peak, crest_factor, envelope_variance;
  - `span: 'levels'` (store at the finest level, roll up by code >> shift) or `'time'` (per level: "the level IS the band", D84).
- **The lookup** is `MeasurementQuery` → `db.measurements.query` (`database.ts:205, 1173`; `repository.ts:330`; `GET /api/measurements`):
  - SQL over SCALAR `selection_measurement` rows;
  - it rolls up (`code >> shift`) under the measure's reduction and derives;
  - it thresholds with `minValue` / `maxValue` and returns `{code, value, weight, members}`.
  - It does not read array-backed measurements.
- **Measurement functions:**
  - `measureOverPartition` (`:208`): spatial partitions;
  - `reduceTiles` (`:278`) and `tileMeasurements` (`:306`): contiguous time windows; min, max, sum, sumSq and count per tile;
  - `rollup.ts`: `rollTime` (`:50`) and `derive` (`:69`).
  - None is called from the Studio app (`studio/src`). The time-axis measurement path is backend and scripts only.
- **Array-backed measurements** (D96, the envelope [channels × windows × 2], `envelope.ts:25-60`): `Field.envelope` (`field.ts:115`) finds them.
  Nothing outside tests calls it or reads the envelope's values.
- **Statistics over a whole recording** (median, MAD, a percentile): no stored ones anywhere. The `measure` dictionary has none. Where they are used
  they are computed fresh, in memory, at detection time (`curlCoreBursts.ts` `spindleEnvelopeDetail`, `percentileLinear`). The display ranges
  (`gpu/range.ts`) are transient.

nxr-neuro has ported this layer: `nxr::partition` (reduceTiles, measure), `io/measures.h` (the registry, `write_measurements`, `query_measurements` with
roll-up and derived measures) and `measure over time`.

## 3. Deriving a time labelling from measurements: the references

- **The Studio:**
  - **`searchAtlas`** (`backend/src/entities/atlas.ts:158-218`, D137) is the ONE implementation that turns measurements into a time labelling:
    1. read a spatio-temporal atlas array;
    2. roll it up (`rollSpace` / `rollTime`), optionally normalised;
    3. keep the cells above or below a value (or the top n);
    4. map each hit to its time interval;
    5. sort and merge the intervals;
    6. return a NEW transient `spans` Selection (code null, no dictionary, `params_json.search` = the query) with its rows, plus the chunks under it.
    - It is in memory ("Save keeps it"), and its callers are tests only.
  - The scalar lookup's `minValue` threshold ("tiles where this field's mean power exceeds T", `measurements.ts:1-17`) is designed for this. Nothing
    turns its result into a Selection.
  - `curlCoreBursts.ts` (`spindleEnvelopeDetail :137`, `burstStages :259`, `burstMask :277`) is a CPU port of nxr's vortex gates (below). Nothing calls
    it outside tests, and it never makes a Selection.
- **Rheome** (each detector returns in memory unless stated):
  - **`detect.spindle`** (`+detect/spindle.m`) is the general envelope detector:
    - input [channels × time] at fs, resampled to 100 Hz;
    - band-pass 11–16 Hz (IIR order 6, zero phase) and |Hilbert| (or `envelope=true`: the input is already an envelope);
    - a 0.2 s centred moving mean;
    - PER CHANNEL thresholds over a baseline mask or the whole input: median + 3.0 robust SD (1.4826 MAD) to enter, + 2.5 to extend (hysteresis);
    - events merged when the gap is ≤ 0.1 s, then kept when 0.5–3.0 s long;
    - out: a table (channel, start/end sample and s, duration, peak amplitude, peak frequency, peak time).
    - It is used by `labelalpha` and by lab scripts. No golden is stored; `tests/tSpindle.m` uses synthetic data.
  - **`detect.risefall`** (`+detect/risefall.m`) LOCATES one candidate episode by a double-sigmoid fit:
    - out: tOn, tOff, the rise and fall τ, r², and the build-up, plateau and decay intervals;
    - the callers apply r² ≥ 0.7;
    - default model "linear".
  - **`flow.labelalpha`** (`+flow/labelalpha.m`) labels alpha in SOURCE space:
    - its episode family runs `detect.spindle` in envelope mode on the peak depth-7 tile amplitude per hemisphere, at 1.5 / 1.0 robust SD,
      min 0.4 s, no max, then `risefall`;
    - its rows go to Rheome's select-store label sidecar;
    - it also has a class family (on/off by a GMM, `detect.onstate`) and a trajectory family (paths), which are not time-axis events.
  - **`cores.burst*`:** a MATLAB port of nxr-analysis's vortex gates, for goldens only (callers: `tests/tVortexSignal.m`).
  - **`scale.cleanspan`** (`+scale/cleanspan.m`) is the one automatic BAD-SPAN detector:
    - 0.5 s blocks;
    - the median over channels of each block's RMS about the channel median;
    - a block is bad above 3 × the median block level, broadband or below 4 Hz;
    - bad blocks are dilated by one block, and the recording's edges trimmed within 10 s;
    - out: a per-sample mask.
  - **Slow waves:** only inside a research measure (`scale/measure_slowosc.m:321`, Massimini 2004: zero crossings 0.3–1.0 s, negative peak < −80 µV,
    peak-to-peak > 140 µV, ≥ 5 channels within 0.2 s). It is not a reusable function.
  - **Not present:** sleep staging, K-complexes, any Brainstorm-event export.
- **nxr:**
  - **`detect/sleep_events.py`** is the one general detector (spindles and slow waves of a night of EEG):
    - it works at 100 Hz after `decimate` (an average reference, a 40 Hz low-pass, ÷ 10);
    - it uses the sleep stage per sample (`stage_track`, from imported scored stages).
    - Slow waves:
      - band-pass 0.5–4 Hz;
      - negative half-waves 0.25–1.0 s starting in N2/N3;
      - depth ≤ the 20th PERCENTILE of the N3 troughs pooled over all channels and the night;
      - channels merged within 0.4 s of the first trough.
    - Spindles:
      - |Hilbert| of 11–16 Hz with a 0.2 s moving mean;
      - lo = 4 × and hi = 6 × the MEDIAN over channels at NREM samples;
      - runs ≥ lo in NREM, 0.5–3.0 s long, reaching hi;
      - the union over channels;
      - a spectral purity check.
    - Out: npz files only, no Selection, and its doc says it was never compared with Rheome's.
  - **`detect/vortex.py`** `burst_stages` / `spindle_envelope` / `risefall` are per-frame GATES (36 frames a cycle) for the phase-resolved vortex-core
    analysis. They are not event detectors. `burst_mask` is dead code (no caller).
  - **nxr writes no detected events as Selections:** only the importers call `time_spans`.

## 4. What the audit finds for nxr-neuro (facts that bound a design; no design here)

1. **The output form is settled by the model and the importer:**
   - a `spans` Selection on the time line;
   - code = the event type through a dictionary;
   - per-occurrence channels in `params_json.channels`.
2. **Bad segments have three forms in the references:**
   - imported (Brainstorm, MNE, BIDS) as `<rec>_bad_segments`;
   - detected by Brainstorm and imported (`import_bad_segments`);
   - detected by Rheome's `cleanspan`, from block RMS about the median, in 0.5 s blocks: a WINDOWED measurement (RMS per block) thresholded by a
     whole-recording statistic (the median over blocks).
3. **The Studio's only measurement-to-labelling step is `searchAtlas`:** threshold windowed measurements, take each hit's tile interval, merge, and
   return a spans Selection. Its threshold is a value the caller gives.
4. **Every detector whose threshold depends on the recording computes the statistic over a whole span:**
   - Rheome `detect.spindle` / `labelalpha`: the median and MAD of the smoothed envelope;
   - nxr `sleep_events`: a percentile of troughs, a multiple of the median;
   - `cleanspan`: the median of the block levels.
   - Most compute it per SAMPLE of a smoothed envelope at 100 Hz; `cleanspan` on its 0.5 s blocks.
   - Neither the Studio nor Rheome stores these statistics.
5. **The references disagree on thresholds and units:**
   - Rheome's spindle: 3.0 / 2.5 robust SD per channel;
   - its alpha episodes: 1.5 / 1.0;
   - nxr's spindles: 4 × / 6 × the median over channels;
   - nxr's slow waves: a percentile.
   - Durations are in seconds (Rheome, nxr sleep) or in cycles (the vortex gates).
   - A port names its reference per detector; there is no single "the" detector.
6. **No reference stores a golden for `detect.spindle`, `risefall` or `labelalpha`.**
   - Rheome's `tests/goldens/vortex-signal/` (synthetic, from nxr-analysis) covers only the vortex gates.
   - nxr's `sleep_events` has no golden.
7. **nxr-neuro already has the pieces for the sampled layer:**
   - windowed measurements over tiles (`measure over time`, `nxr::partition::reduceTiles`);
   - the lookup (`query_measurements`);
   - writing spans Selections (`nxr::bst::time_spans`);
   - the pure span blocks (`nxr::intervals`: crossings with hysteresis, mergeSpans, keepSpans, halfWaves, median, mad).
   - Missing: a step that turns thresholded measurements (or a smoothed envelope) into a spans Selection; and the place a whole-recording statistic
     lives (the statistics layer, open: decision 1 of the event spec).
