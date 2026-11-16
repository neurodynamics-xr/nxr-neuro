# Event detection: building blocks chained into detection pipelines (spec, 2026-11-13, revision 3: the runner decisions of 2026-11-13 below; the rest a DRAFT for the maintainers)

## Why

Everything built so far averages, over a span (`phasegradient`), a window (`cyclepattern`) or a member of a partition (`measure`). The patterns the
lab studies are also single events: a burst of a rhythm, a rotational core that forms, moves and dies within a few cycles, a phase singularity. An
average cannot show their number, timing, lifetime, path or speed.

nxr (Python) detects these, but each detector is one module that mixes its own operator, thresholds, components, tracking and summaries (e.g.
`detect/vortex.py`). The the maintainers' direction (2026-11-13): organise detection better than nxr. Detection is BUILDING BLOCKS (most already built) chained
into PIPELINES, and the blocks are organised by what they produce:
- detecting time windows of interest in a time series is one kind of work;
- detecting spatial features on the cortex, and tracking them over time, is another.

nxr's detectors are then pipeline DEFINITIONS that reproduce nxr's results (goldens against nxr), not ported modules. A new detector is a new
definition, not new code.

## The four families

### A. Temporal events: time windows of interest in a time series

- **In:** any time series: sensor channels, a source or region signal, a mode coefficient, an amplitude. The method never knows where it came from.
- **Out:** a TIME SELECTION, spans on the recording's line (`type = spans`, as the importer stores bad segments), each with its attributes (peak,
  duration, the channel that found it).
- **Examples:** bursts of a rhythm (nxr `burst_mask`, `burst_stages`), spindles and slow waves (nxr `sleep_events`), artefact spans, spikes.

| block | does | status |
|---|---|---|
| `band_filter`, `analytic`, `envelope` | the band, its analytic signal, its amplitude | have (registered steps) |
| `power`, a mean over channels | |z|²; the mean over a channel partition (a measurement) | have |
| smoothing | a moving average or low-pass of the amplitude (nxr: one or two cycles) | have as filters; a boxcar `smooth` step to add |
| span-wide statistics | percentile, median, robust SD (1.4826 · MAD) of a series over the span: MEASUREMENTS over a time partition (nxr's 4096-bin log histogram) | to add: `percentile`, `median`, `mad` measures |
| `crossings` | spans where x ≥ an entry threshold, each extended while x ≥ an exit threshold (hysteresis; entry = exit gives a plain threshold) | to add |
| `merge_spans` | spans closer than a gap joined | to add |
| `keep_spans` | spans kept by duration (min, max) | to add |
| `merge_channels` | per-channel spans joined across channels (nxr sleep events: troughs within 0.4 s) | to add |
| `half_waves` | negative half-waves between zero crossings, by duration and depth (nxr slow waves) | to add |
| `rise_fall` | the double-sigmoid fit of a span's amplitude: onset, offset, rise and fall times, r² (nxr `risefall`, from Rheome) | to add (describes a span: family D) |

### B. Spatial features on the cortex, frame by frame

- **In:** a field on the surface at one frame: the curl of the band's current, its stream function or normal current, a complex field's phase.
- **Out:** features of that frame, with attributes: points or patches with sense, centre, area, peak.

| block | does | status |
|---|---|---|
| the field | kernel, full forms, spatial band, forms | have |
| `threshold_mask` | vertices where sense · x ≥ a threshold (a span-wide percentile: a measurement, as in A) | to add |
| `components` | connected patches of a masked vertex graph, each with area, area-weighted centre, peak (nxr `_core_patches`) | to add |
| `extrema` | vertex maxima / minima above a prominence (nxr `flow/stream.extrema`) | to add |
| `winding_faces` | faces whose phase winds ±2π, and where the interpolant vanishes (nxr `cycle_cores.detect`, `stream.phase_singularities`) | to add |

### C. Tracking: features linked across frames

- **In:** B's features per frame. **Out:** tracks. Independent of the feature: the same step tracks cores, singularities or extrema.
- **Blocks:** `link_nearest` (greedy, same sense, ≤ a distance, no gap: nxr `_track_stats`), `link_assignment` (Hungarian: nxr `cycle_cores.track`),
  `link_overlap` (by area overlap, with splits and merges: nxr `curl_atoms`). Open tracks are an accumulator carried across chunk edges.

### D. Describing events

- Of a span: duration, peak, the rise/fall fit. Of a track: lifetime, path length, net displacement, speed, straightness; recurrence one cycle later;
  standing or travelling (nxr `core_phase`: σ = |W c²| / W|c|², the crest velocity).
- All are MEASUREMENTS over the event (a span, a track) through the existing measure system, never code inside a detector.

## How the families compose

A's output is a time Selection, and a time Selection is what restricts B and C. "Vortex cores within alpha bursts" (nxr `vortex.py`) is two pipelines:
1. bursts (A): sensors → alpha band → analytic → power → mean over channels → smooth → threshold (the 75th percentile over the span) → crossings →
   keep ≥ 3 cycles → spans;
2. cores (B, C, D): the source current → its curl in a spatial band → threshold (the 98th percentile of |curl| over the BURST spans) → components →
   link (≤ 15 mm, same sense) → tracks → measures.

The burst spans are the only thing passed between them, by name. Either runs alone: bursts with no cortex; cores over a whole span with no bursts.

## What the runner needs (DECISIONS for the maintainers)

1. **Two passes over the span.** A span-wide threshold is a measurement made before the detection can run: pass 1 measures (the percentile), pass 2
   detects with it. The runner has one chunk loop. Proposal: a definition may have a `measure` section whose loop runs first; its results are
   constants of the main loop. The memory held is the measures' accumulators (a histogram per series).
2. **A time Selection as an output.** The runner stores Fields. Proposal: an output of kind `spans` is stored as a Selection on the recording's time
   line (with each span's attributes in its rows' measurements), as the importer stores bad segments.
3. **Features and tracks (B, C) as stored data.** Proposal, with no io change: an event table is a `points` manifold (one point per feature), its
   columns Fields on it (frame, time, sense, centre, area, track), its tracks a Selection (`set`, code = the track). The alternative, a new io node
   kind, is an io change.
4. **The time partition.** Whether spans, windows and summaries use the canonical dyadic time tiles from the start (the vocabulary question of
   2026-11-13).
   **Answered: yes.** Time is one dyadic tile partition. Spans and windows are
   runs of whole tiles of 2^k samples on the recording's line, seconds snap to the nearest tile boundary, and the snapped extent goes in the provenance.
   The rework of the existing verbs is the amendment of 2026-11-13 to `2026-11-13-cycle-pattern.md`. Family A's spans are stored on the same grid.

## The runner decisions

### Decision 1: global statistics are stored nodes, referenced by the windowed pipelines

**Decision:** the architecture is a pipeline/compute graph of modular steps; global statistics are stored nodes the windowed pipelines refer to.

**Amended (2026-11-13), the same day:** the distribution is a statistical measure, for which an output policy is still to be set; it lives in the statistics layer.

So the layers are:
- **Fields:** the samples.
- **Selection/measurement:** sampled measurements per member of a partition (a tile, a window, a parcel); a table of values.
- **Statistics:** produced from the samples or the measurements, over a whole recording or a span: a distribution, a percentile, a median and MAD, and
  later tests across subjects.

What decision 1 already fixes:
- A global statistic is its own node, made once and referenced by the windowed pipelines.
- A pipeline reads it if it is stored, and computes and stores it if not.
- It never runs as a pass inside a detector.

What is OPEN: the statistics module and its storage (where a statistic and its provenance are stored, and how a pipeline names it). Their own spec,
for the maintainers. Neither reference has such a layer to port: the Studio's model has no statistics table, and nxr's statistics are analysis helpers
(`analyses/_stats.py`) whose results go to npz/json files.

The distribution's arithmetic is settled: `nxr::histogram`, nxr's `_core_threshold` ported (4096 log bins, the range from the first chunk, values above
it clipped into the top bin). Its counts are additive over chunks, and percentile, median and MAD are read off them, exact to one bin (0.34 %). Its
storage waits for the statistics spec.

### Decision 2: detected spans are a `spans` Selection, in the importer's form of events

This is corrected by the audit (`docs/audits/2026-11-13-time-event-labelling.md`). The first version said "code = the span's ordinal". The model (D143)
and the importer (`nxr::bst::time_spans`) use code = the event TYPE.
- **The form:** a spans Selection on the recording's time line, one `selection_span` row per event, [start, stop) in seconds.
  - `code` = the event type, through a dictionary named after the Selection.
  - The channel that found an event goes in `params_json.channels` {ordinal: [channels]}, as the importer stores Brainstorm's per-event channels.
  - How it was made goes in `params_json` (the Studio's `searchAtlas` records its query there).
- **An event's attributes** (peak, mean amplitude) are the measurement rows of the tiles it covers, read through the lookup, and not stored again.
- **Built in** `docs/specs/2026-11-13-time-events.md` (`band_bursts`).

### Decision 3 (later, before families B and C): features and tracks

- **Neither reference stores them.** nxr keeps cores and tracks in memory and writes summaries and density maps (`.npz`/`.json`). The Studio tracks in
  memory and exports CSV.
- **Storing them is therefore new design.** The recommendation stays the one above (a `points` manifold, Fields as columns, a track a Selection), because
  family D's track measures need a Selection to measure over.
- To be decided by the maintainers when B starts.

## Order (each a PR, each block with a unit test, each nxr detector with a golden against nxr)

1. **A's blocks** as pure functions (`compute/include/nxr/intervals.h`): `crossings`, `mergeSpans`, `keepSpans`, `mergeChannels`, `halfWaves`,
   `riseFall`; the robust statistics (`percentile` by log histogram, `median`, `mad`) as measures. Unit tests on synthetic series. **(started)**
2. The runner's two passes and the `spans` output (decisions 1, 2); the steps registered.
3. **Bursts** as a definition, golden against nxr `burst_mask` and `burst_stages`.
4. B's and C's blocks, decision 3, then cores and singularities as definitions (goldens against nxr `core_stats`, `cycle_cores`).
5. D: track measures, `core_phase`'s standing/travelling.
6. Surrogates (nxr `lagfree_real`, `envelope_matched_null`) as a step that writes a surrogate recording any pipeline can read.
7. Sleep slow waves and spindles as definitions of A's blocks.

## Checklist (the architecture policy)

- Each block is a pure function of arrays (a series, a mask, a mesh's adjacency), callable with data from anywhere.
- A detector is a definition: it runs on any stored series or field of the right kind, by name.
- Every loop over samples or chunks is the chunk service's; tracking and open spans are accumulators carried across chunks.
- Every threshold that spans time is a measurement over a time partition, never a loop of its own.
