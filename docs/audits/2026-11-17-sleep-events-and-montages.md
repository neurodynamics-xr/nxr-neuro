# Audit, 2026-11-17: spindles, slow waves and standard electrode montages (read-only)

**Purpose.** The maintainers decided (2026-11-17):
- spindle and slow-wave detectors are ported from Rheome, in the burst family or one like it;
- standard electrode montages are shipped on the templates;
- there is no native BEM model;
- there is no sleep-stage detection.

The project audits before it ports. This audit records what the references do, how they map onto nxr-neuro's burst family and templates, and
an ordered PR plan. It is not a spec. Every claim about code was read in the code.

- **Read:** Rheome (`+rheome/+detect/spindle.m`, `onstate.m`, `risefall.m`, `onsetspread.m`; `+rheome/+scale/measure_slowosc.m`, `groupslowosc.m`,
  `run.m`; `+rheome/+cores/burstfit.m`, `boxsmooth.m`; `tests/tSpindle.m`, `tests/tScaleSlowOsc.m`). Rheome is MIT-licensed (its `LICENSE`).
- **Read:** nxr-neuro at main (39cc5ae): `ARCHITECTURE.md`, `CONTRIBUTING.md`, `pipelines/definitions/band_bursts.yaml` and `global_bursts.yaml`,
  `compute/include/nxr/intervals.h`, `compute/src/intervals.cpp`, `docs/reference.md`, the specs `2026-11-13-time-events.md` and
  `2026-11-13-cortical-events.md`, the audits `2026-11-13-time-event-labelling.md` and `2026-11-17-format-mapping-and-coordinates.md`,
  `io/convert/manifest/template_sources.json`.
- **Read:** MNE-Python 1.13.2 (`mne/channels/_standard_montage_utils.py`, `mne/channels/data/montages/` and its `README.md`), and the file list
  of Brainstorm's default EEG caps.
- No dataset content was read.

---

## 1. Spindles

### 1.1 Rheome's algorithm, `rheome.detect.spindle(X, fs, opts)`

The input is [channels × time] at `fs`. The output is a table with one row per event and channel: `channel`, `startSample`, `endSample`,
`startSec`, `durationSec`, `peakAmp`, `peakFreq` and `peakSec`. Samples are on the input's own grid.

1. **Decimate** to `workRate` (100 Hz), using MATLAB's polyphase `resample` at the rate `rat(workRate/fs)`. Every reported index is mapped back to
   the original grid as `round((i−1)·fs/fsw) + 1`. The baseline mask is resampled too, as `> 0.5`.
2. **Band-pass** `band` [11 16] Hz. The filter is `designfilt('bandpassiir', FilterOrder 6)` (Butterworth, half-power edges), run with `filtfilt`
   (zero phase). It refuses a band whose upper edge is not under 0.45·fs, and an unstable filter.
3. **Envelope:** `abs(hilbert(·))` over the whole input.
   - With `envelope = true` the input already IS an envelope: the band-pass and Hilbert steps are skipped, the input is clipped at 0, and
     `peakFreq` is NaN.
   - Rheome's alpha labelling uses this mode on source-space tile envelopes.
4. **Smooth:** `movmean` over `round(smoothSec·fsw)` samples (0.2 s, so 20 samples). MATLAB centres an even window as [i−10, i+9] and shrinks
   it at the ends.
5. **Per channel:** the baseline is the median and the robust SD (1.4826 · MAD) of the smoothed envelope.
   - They are taken over `baseline` samples, or over the whole input. They are never taken per segment.
   - The file explains why: a per-segment baseline makes wake look like N2, so the detector ends up measuring its own threshold.
   - A channel whose robust SD is 0 is skipped.
6. **Hysteresis:** runs where the envelope is at least the median + `loSD`·SD (2.5). A run is kept only if it reaches the median + `hiSD`·SD (3.0).
7. **Merge:** after the high-threshold gate, runs are joined when `start(next) − last(previous) ≤ round(mergeGap·fsw)` (0.1 s).
8. **Gate on duration:** a run is kept when its length (in work-rate samples, divided by `fsw`) is between `minDur` and `maxDur` (0.5 s and 3.0 s).
9. **Per event:**
   - `peakAmp`: the maximum of the smoothed envelope;
   - `peakSec`: the time of that maximum;
   - `peakFreq`: the arg-max, inside the band, of |FFT|² of the Hann-windowed band-passed segment. The FFT is zero-padded to
     `2^nextpow2(max(n, 4·fsw))`, so the bins are 100/512 ≈ 0.2 Hz wide at 100 Hz.
10. Events are sorted by start time, then by channel.

**Defaults:** band [11 16] Hz, `hiSD` 3.0, `loSD` 2.5, `mergeGap` 0.1 s, `minDur` 0.5 s, `maxDur` 3.0 s, `smoothSec` 0.2 s, `workRate` 100 Hz,
baseline = the whole input. The file records how the thresholds were chosen. On 5 minutes of narrow-band noise with 21 planted spindles at
2.5 × background:
- sensitivity was 21/21 at every setting tried;
- `loSD` 2.5 gave 0 false events a minute, where `loSD` 1.5 gave 0.4;
- raising `hiSD` barely helps, because a noise excursion that reaches it is still extended down to `loSD`.

**Its published reference: none.** The file calls it "the conventional envelope detector" (band-pass, envelope, a high threshold extended to a low
one, duration gating, merging) and cites no paper.
- Its robust baseline (median + k · MAD instead of a percentile or a multiple of the mean) is Rheome's own choice. It is argued in the file and
  tested on synthetic data (`tests/tSpindle.m`).
- Under the method rule, the reference is therefore **Rheome's function itself** (Rheome is one of the named references). No literature
  detector is claimed (open question Q1).

**Limits the file states** (each goes into `producer_json.warnings` and the method's section of `docs/reference.md`):
- the reported duration depends on amplitude near the gate (a loud 0.2 s burst can read as 0.8 s);
- it fires on eye movements and muscle with plausible durations, and only the topography (central-parietal against frontopolar) tells them apart;
- the baseline must be shared by every stage that is compared.

**Related Rheome functions:**
- **`detect.risefall`** fits a double sigmoid to a candidate episode, giving half-rise and half-fall times, τ and r². It is already ported:
  `nxr::risefall` and the `risefall`, `fit_gate` and `burst_stages` steps of `global_bursts`, with a golden against Rheome.
- **`cores.burstfit` and `cores.boxsmooth`** are nxr's quadrature version of that fit and its zero-padded moving mean. They are already ported as
  risefall's `quadrature` model and the `burst_stages` golden.
- **`detect.onstate`** sets an on/off threshold from a two-component Gaussian mixture of log amplitude. Its own documentation says it is for
  rhythms that are NOT sparse, such as alpha, where a median-based threshold sits inside the rhythm. Spindles are sparse, so it is not part of
  this port. It could become a named threshold option of the burst family later.
- **`detect.onsetspread`** measures how a burst's onset (each tile's half-rise time) spreads across cortical tiles, as a slowness in s/m. It is a
  WHERE / travel step for later (section 3), not a detector.

### 1.2 How it maps onto the burst family

`band_bursts` and `global_bursts` already use Rheome's rule, through `labelalpha` → `detect.spindle` in envelope mode at 1.5 / 1.0 SD,
0.4 s and no maximum. The compute blocks are the same as the spindle detector's:

| Rheome step | nxr-neuro today | status |
|---|---|---|
| hysteresis (≥ lo, reaching hi) | `nxr::intervals::crossings(entry, exit)`: same rule, same inclusivity (`x ≥ exit`, any `x ≥ entry`) | exists |
| merge, `start − last ≤ g` (last sample inclusive) | `mergeSpans(gap)`: `start − stop < gap` (stop exclusive). The same condition | exists |
| duration gate [min, max] | `keepSpans(minLength, maxLength)` | exists |
| median, 1.4826 · MAD | `nxr::intervals::median`, `mad`; the `summarize` step `[median, mad]` stores them as a statistic | exists |
| per channel, or one trace | `detect_spans` `combine: each` / `mean` | exists |
| rise and fall refinement | `risefall`, `fit_gate`, `burst_stages` | exists (optional for spindles) |
| roll-up of per-event measures | `rollup_events` (amplitude, envelope_max, duration, interval_next) | exists for events made of tiles |
| band-pass: Butterworth IIR of order N, zero phase | `nxr::filter` has a forward-backward Butterworth for `bandstop` only. The band-pass default is Brainstorm's Kaiser FIR | **new named option** of `band_filter` |
| decimation to the work rate | the time tower (`nxr::tower`, frames) samples a band-limited field. No decimating filter step | **new** (see Q4) |
| `movmean` of 0.2 s (MATLAB's even-window convention, shrinking at the ends) | none on samples. `band_bursts` uses the tile as its smoothing | **new** (a pure function; `boxsmooth`'s zero-padded convention is the other named variant) |
| detection on a SAMPLE series | `detect_spans` reads TILE measurements. `global_bursts` `collect`s a sample series, but only for `risefall` | **new**: detection on a collected or stored sample trace |
| median / MAD over all samples of a long span | in memory over tiles | **new**: exact median over samples with bounded memory (Q6) |
| `peakFreq` (Hann, zero-padded FFT, arg-max in band) | none | **new measurement function** `peak_frequency` |
| `peakSec` | `Span::peak` is kept by `crossings` and `mergeSpans`, but not stored | **new measurement** `peak_time` |

**The band and D84.** `band_bursts` measures a band's amplitude on the tiles of the constant-Q level OF that band (D84: "the level IS the band").
[11 16] Hz is no level: the levels are octaves at the hertz anchor, 8–16 and 16–32 Hz.
- Rheome detects on a smoothed sample envelope, not on tiles. The faithful port therefore detects on SAMPLES at the work rate, and the band is a
  filter parameter, not a level. D84 is then not involved (Q5).
- The tiles come back only as the partition the per-event measurements are taken over: the events themselves.

**The definition `sleep_spindles`** (new; a sketch, not a spec):

```
inputs:  recording (each), bad (optional), within (optional: a spans Selection the detection and the baseline are restricted to,
         e.g. imported scored N2/N3 stages; never detected here)
chunks over recording:
  band_filter  (iir: butterworth, order 6, [11 16] Hz, zero phase)       new option
  downsample     (to the work rate, a dyadic divisor of fs: Q4)             new node
  envelope     (|analytic|)                                               exists
  smooth       (moving mean 0.2 s, movmean convention)                    new node
  collect / store the work-rate trace [channels × samples]                exists (collect) / new (trace as a Field)
after:
  summarize    (median, mad over the good samples of the span, per channel)    exists, over samples: Q6
  detect_spans (on samples: enter 3.0, exit 2.5, merge 0.1 s, min 0.5 s, max 3.0 s, combine each)   exists, sample input new
  join_rows    (per-channel spans → global events: Q3)                    new node
outputs: spindles (spans, type `spindle`), the baseline (statistic), per-event measurements (section 1.3)
```

### 1.3 What is stored

- **The events:** a `spans` Selection on the recording's time line, in the importer's form.
  - Name: `sleep_spindles.spindles.<recording>`.
  - `code` = the type `spindle`, through its dictionary.
  - Each event's channels go in `params_json.channels`.
  - `params_json` also holds the pipeline, its parameters, the thresholds used per channel and the cache key.
  - This is the same form as the bursts. Every reader of `<rec>_events` reads it.
- **Per event:** measurements over the events Selection, as a partition (the cardinal rule). Each measurement function reduces WITHIN one event.
  Stored array-backed [rows × events × 2], as the burst measures are:

| measure | within an event | reference |
|---|---|---|
| `duration` | stop − start (s) | Rheome `durationSec` |
| `amplitude` | mean smoothed envelope | (the bursts') |
| `envelope_max` | max smoothed envelope | Rheome `peakAmp` |
| `peak_time` (new) | time of that max (s, the recording's clock) | Rheome `peakSec` |
| `peak_frequency` (new) | arg-max of the Hann-windowed, zero-padded spectrum of the band-passed segment, inside the band (Hz) | Rheome `peakFreq` |
| `interval_next` | the next event's start − this stop | (the bursts') |
| `t_on`, `t_off`, `tau_rise`, `tau_fall`, `fit_r2`, … | `risefall`, optional | Rheome `detect.risefall` |

- **Derived at read, never stored:**
  - symmetry = (`peak_time` − start) / `duration`;
  - number of cycles = `duration` × `peak_frequency`;
  - density = count / good seconds × 60;
  - occupancy.
  The statistics across events record the span's good seconds, as `band_bursts` does.
- **Across events:** `summarize` (count, sum, median, mean, sd, percentiles), as for the bursts.
- **Topography (the sensor-level WHERE):** each global event is measured on EVERY channel row, not only on the rows that detected it, so the
  rows give the spatial distribution. This is the check the reference asks for against artefacts.
- **New measure codes** come from nxr-neuro's reserved range (1000–1999; 1126 is the last in use).
- **Computing the per-event measures:** they need the samples inside each event. Two routes:
  - **(a)** in the after-steps, from the collected work-rate trace and the collected band-passed signal;
  - **(b)** a second pass over the recording measured over the events Selection (`measure` over a spans Selection: the generic event-measures
    pass that `docs/reference.md` lists as not built).

  Route (b) is recommended. It also measures imported events and needs no extra memory (Q7).

---

## 2. Slow waves / slow oscillations

### 2.1 Rheome's algorithm (`measure_slowosc.m`: the preparation at lines 156–159, the local function `i_detect` at 321–345)

**Where it lives.** Slow-wave detection exists in Rheome only inside a research measurement (`rheome.scale.measure_slowosc`), as a LOCAL function.
- It is not callable from outside the file.
- No other slow-wave detector is in Rheome. `groupslowosc` reads its output, and `eventlocked`, `sleepeeg` and `importeeg` only prepare or consume
  it.
- K-complexes are not present.

**Its reference:** Massimini M, Huber R, Ferrarelli F, Hill S, Tononi G (2004) The sleep slow oscillation as a traveling wave. *J Neurosci*
24:6862–6870. The criteria are theirs, and Rheome applies them unchanged.

1. **Prepare:**
   - the selected EEG channels under the reference operator `S.Ref` (Rheome uses the average reference);
   - `butter(2, Band/(fs/2))` band-pass [0.1 4] Hz (order 4 in total), run with `filtfilt` separately on each recorded segment.
2. **Per channel**, where `s = sign(y)` with 0 counted as +1:
   - a downward zero crossing `dn`, followed by the first upward one `up`;
   - the half-wave length `(up − dn)/fs` within `HalfWaveS` [0.3 1.0] s;
   - the trough `min y(dn:up)` ≤ `NegUV` (−80 µV);
   - the following positive peak `max y(up : e)` ≥ trough + `P2PUV` (140 µV). Here `e` is the next downward crossing after `up`, or `up + fs`
     samples when there is none;
   - a candidate is (channel, trough sample, trough, peak-to-peak).
3. **Across channels:**
   - candidates are sorted by trough sample;
   - a group is every candidate whose trough falls within `ClusterS` (0.2 s) of the group's FIRST trough. The window is anchored, not chained;
   - the group is an event when at least `MinChannels` (5) distinct channels are in it;
   - a group that falls short is consumed whole: its candidates do not start new groups.
4. **Per event:**
   - `sample` = `round(median(trough samples))`, Rheome's event time;
   - `nChannels`;
   - `negUV` = the minimum trough;
   - `p2pUV` = the maximum peak-to-peak;
   - `seedChannel` = the channel of the earliest trough.
5. **Analysis only (not detection):** the analysis-window fit, the artefact-epoch mask, the random draw of `MaxEvents` and the four direction
   estimators are the research measurement, not the detector. They are not ported with it. The `bad` input and the caller's span cover exclusion.

**Defaults:** band [0.1 4] Hz (Butterworth order 2, zero phase), half-wave 0.3–1.0 s, trough ≤ −80 µV, peak-to-peak ≥ 140 µV, cluster window
0.2 s, at least 5 channels, average reference.

**Its limits, from the file:**
- the thresholds are absolute (µV), so they depend on the reference: a reference that rescales amplitudes changes the number of events;
- on synthetic and template heads, sensor-latency direction also reads two lagged generators as travel (it reads lag, not travel).
  That limit belongs to the later WHERE / travel step.

### 2.2 How it maps onto nxr-neuro

| Rheome step | nxr-neuro today | status |
|---|---|---|
| the downward → upward half-wave, its length, its trough | `nxr::intervals::halfWaves(x, minLength, maxLength)`: start = the first negative sample (`dn + 1`), stop = the first non-negative one (`up + 1`), so length = `up − dn`, as Rheome's; 0 counted as non-negative, as Rheome's `sign` rule; it breaks when the record ends inside a half-wave, as Rheome does | exists. Converting the bounds from seconds to samples (Rheome compares seconds) is checked by the golden |
| trough ≤ −80 µV, the positive peak to the next down-crossing (or +1 s), peak-to-peak ≥ 140 µV | none | **new** pure function (`slowWaveCandidates`, built on `halfWaves`) |
| the anchored cluster window, minimum distinct channels, median time, min / max / seed | none | **new** pure function (`clusterInstants`) |
| Butterworth order 2, [0.1 4] Hz, zero phase | the same new IIR option of `band_filter` as for spindles. A 0.1 Hz edge needs a halo of tens of seconds, more than one chunk at high rates: the chunk service carries it | new option |
| the average reference | the reference operator is the caller's preparation (a re-reference step, or a recording already referenced) | check that a re-reference node exists, or add one |
| per-segment filtering | segments and bad spans are already the chunk service's | exists |

**The definition `slow_waves`** (new; a sketch):

```
inputs:  recording (each; the EEG channels by the `type` Selection), bad (optional), within (optional: imported NREM spans)
chunks over recording:
  reference    (average)                                     preparation
  band_filter  (iir: butterworth, order 2, [0.1 4] Hz)       new option (shared with spindles)
  half_waves   (per channel: slowWaveCandidates)             new node, stateful across chunks (a half-wave can cross a chunk edge)
after:
  cluster      (clusterInstants: 0.2 s, ≥ 5 channels)        new node
outputs: slow_waves (spans, type `slow_wave`), the per-event and per (event × channel) measurements
```

### 2.3 What is stored

- **The events:** a `spans` Selection of type `slow_wave` on the time line, in the importer's form, with the participating channels in
  `params_json.channels`.
  - Rheome's event is an instant (the median trough).
  - Recommended span: from the earliest downward crossing to the latest upward crossing among the event's half-waves, so the event covers its
    negative phase. The instant is kept as a measurement.
  - A named option `extent: instant` stores start = stop, Rheome's own form (Q9).
- **Per event** (measurements over the events partition):
  - `trough_time` (Rheome's `sample`, as seconds);
  - `n_channels`;
  - `trough` (the minimum; Rheome's `negUV`, stored in V);
  - `peak_to_peak` (the maximum; `p2pUV`);
  - `seed_channel` (a channel index; reduction none).
- **Per (event × channel)**, the sensor-level WHERE, NaN on channels with no half-wave in the event:
  - each channel's trough, trough time, peak-to-peak and half-wave length.

  The per-channel trough times are what a latency-gradient (travel) reading uses later. That reading is not part of this port.
- **Derived at read:**
  - the latency spread (max − min of the channel trough times);
  - the event rate;
  - the slopes (peak-to-peak / the trough-to-peak time), if wanted. They are not in Rheome; nxr-neuro may add them as its own measures.
- **Across events:** `summarize`, as for the bursts.

---

## 3. Per channel or global

The decided rule (2026-11-13, in `2026-11-13-cortical-events.md`) is that bursts are global. Time detection says roughly WHEN, and the cortical
analysis says WHERE. A mean over channels was refused, because it washes out local increases.

- **Slow waves already fit the rule.** Rheome's detector is per channel only at the candidate stage. The EVENT is the cluster across channels,
  a global WHEN with its channel set.
  - The port stores global events only.
  - The per-channel candidates are an intermediate. Their values are kept as per (event × channel) measurements, which are the sensor-level
    WHERE.
- **Spindles do not fit it as Rheome writes them.** `detect.spindle` returns per-channel events.
  - Recommended: run the detection per channel exactly as Rheome does. The golden is tested on this intermediate, which keeps the method rule's
    parity.
  - Then JOIN the rows into global events:
    - per-channel spans that overlap, or lie within `merge_s` of each other, become one event;
    - its channels are recorded;
    - an event is kept when at least `min_channels` rows found it. The default is 1, the union, which is nxr's `sleep_events` rule. Rheome sets
      no rule for this.
  - The stored Selection is the global one.
  - Storing the per-channel spans, Rheome's own output form, is a named option (`events: per_channel`) for parity reports and for users who need
    it (Q3).
  - The other named global route is the one `global_bursts` already takes: detect on one trace that is the MAX over rows. That is the max over
    channels at sensor level, or the max over cortical tiles in source space, through Rheome's envelope mode, which is how `labelalpha` uses it.
    It needs no join step. It is not the reference's output, so it is an option, not the default.
- **WHERE beyond the sensors**, a later step with a spec of its own:
  - per global event, the band envelope of the source estimate measured per spatial tile WITHIN the event (a measurement over the joint
    partition of events × spatial tiles);
  - then the existing hysteresis on the tile graph (`detect_regions`, as phase 1 of the cortical-events spec);
  - for slow waves, how their onset travels (Rheome's `detect.onsetspread`, slowness in s/m), with the lagged-generator null Rheome documents.
  - This keeps the spindle band out of D84, because the cortical WHERE measures within events, not on constant-Q cells. `global_bursts` with
    its octave levels stays the cortical WHEN route for octave bands.

---

## 4. Montages

### 4.1 What exists to ship from

- **MNE-Python 1.13.2** (BSD-3-Clause) has 34 built-in montages. Their data files and where they come from, according to MNE's
  `montages/README.md` and the reader code:

| MNE montage(s) | data file | where it comes from | terms of the data | frame as MNE reads it |
|---|---|---|---|---|
| `spherical_1005`, `_1010`, `_1020` (344 / 70 / 21 + 3 fiducials) | `.tsv` | the `eeg_positions` package (data `Nz-T10-Iz-T9/standard_10xx_3D.tsv`): the 10-05 rule computed on a unit sphere | MIT (`eeg_positions`), to be confirmed at the PR | `unknown` (unit sphere) |
| `fsaverage_1005`, `_1010`, `_1020` (335 / 70 / 21 + 3) | `.tsv` (metres) | computed by an MNE developer on fsaverage's scalp (linked in the source as a gist) | BSD-3 (MNE), the positions placed on a FreeSurfer-derived head | `unknown` in MNE; the numbers are fsaverage's surface RAS (m) |
| `colin27_1005`, `_1020`, `_alphabetic`, `_postfixed`, `_prefixed`, `_primed` | `.elc` | FieldTrip `template/electrode/standard_*.elc`: the 10-05 positions fitted on the Colin27 head (Oostenveld and Praamstra 2001) | FieldTrip is GPL-3.0. The data files' own terms are not stated | `mri` |
| `easycap-M1`, `M10`, `M43`, `brainproducts-RNP-BA-128` | `.txt` (θ, φ) | FieldTrip and the manufacturers | not stated | `unknown` (sphere) |
| `GSN-HydroCel-*`, `EGI_256` | `.sfp`, `.csd` | FieldTrip (HydroCel) and a CSD-toolbox file (EGI 256) | not stated | `unknown` |
| `biosemi*` | `.txt` (θ, φ) | the manufacturer's tables | not stated | `unknown` (sphere) |
| `mgh60`, `mgh70`, `artinis-*` | `.elc` | MGH caps, fNIRS optodes | not stated | `mri` |

- **Brainstorm** ships 48 cap files on ICBM152 (`defaults/eeg/ICBM152/channel_*.mat`: the 10-20 with 19 channels, the 10-10 with 65, ASA 10-05
  with 343, ANT, BioSemi, BrainProducts, GSN and HydroCel, Neuroscan, …).
  - They are distributed with Brainstorm under the GPL. Their originals (ASA, manufacturers) have terms that are not stated.
  - Under `ARCHITECTURE.md` (no GPL code copied) and the method rule (Brainstorm's outputs only as reference goldens), these files are neither
    shipped nor committed.
  - They may serve as an optional by-hand check, read from the user's own Brainstorm installation and skipped when it is absent.
- **The 10-20, 10-10 and 10-05 systems are RULES, not data:**
  - positions at fixed percentages of arcs between the nasion, the inion and the preauricular points;
  - references: Jasper (1958); Chatrian et al. (1985); Oostenveld R, Praamstra P (2001) The five percent electrode system for high-resolution EEG
    and ERP measurements, *Clin Neurophysiol* 112:713–719; Jurcak V, Tsuzuki D, Dan I (2007) 10/20, 10/10, and 10/5 systems revisited,
    *NeuroImage* 34:1600–1611.
  - Positions COMPUTED by nxr-neuro from a template's own scalp and landmarks carry no third-party data licence beyond the scalp's own.

### 4.2 Recommendation: what to ship

1. **Ship 10-05, with 10-10 and 10-20 as Selections of it.**
   - One points manifold `eeg_1005` per template, computed natively by the 10-05 rule on that template's scalp.
   - `10-10` and `10-20` are members of a `system` family (Selections) on it. They are not separate manifolds: D73 says a type is a Selection,
     never a manifold of its own.
   - The old 10-20 names (T3, T4, T5, T6) are aliases of T7, T8, P7, P8, kept in the manifold's label dictionary.
2. **Do not ship vendor caps** (EGI or HydroCel, BioSemi, EasyCap numbered, ANT, Neuroscan) until each file's terms are confirmed in
   `template_sources.json` (Q11).
   - Users can bring their own cap file through the format readers (`.sfp`, `.elc`, `.csd`, BIDS `electrodes.tsv`, BrainVision coordinates).
   - The same placement step then applies (section 4.4).
3. **References for the goldens** (the generators only call the reference, and nothing is a run-time dependency):
   - on a synthetic SPHERE scalp, the rule must reproduce `eeg_positions` / MNE `spherical_1005` (the same rule on a sphere);
   - on fsaverage, MNE's `fsaverage_1005`, `_1010` and `_1020` are the comparison. That is a by-hand check, because the template's scalp is not
     in the repository.

### 4.3 Placing them on the templates

**Prerequisites:**
- **A scalp surface.** Neither template importer reads one today:
  - The `icbm152` recipe reads `tess_cortex_pial_*.mat`, `subjectimage_T1.mat` and `license.txt`.
  - Brainstorm's ICBM152 folder holds `tess_head.mat`, the scalp. Its terms are the folder's MNI/ICBM152 notice, to be confirmed in the manifest.
  - FreeSurfer's fsaverage has no scalp surface. The candidate is MNE's fsaverage `bem/outer_skin.surf`, derived from fsaverage's MRI under
    FreeSurfer's terms (Q12).
  - **The scalp is stored as a surface for placement only.** It is not a BEM: the decided no-BEM rule stands, and the skull layers are not
    imported.
- **Landmarks:**
  - ICBM152: `subjectimage_T1.mat` holds the SCS fiducials (NAS, LPA, RPA) and the NCS points (AC, PC, IH). There is no inion.
  - fsaverage: MNE's `fsaverage-fiducials.fif` has NAS, LPA and RPA.
  - The inion (Iz) of each template must be defined, from its MNI coordinates or its scalp's posterior midline (Q10).
- **G13** (from the format-mapping audit): fsaverage's surfaces are in a unit-less `fsaverage` key. Montage positions must be in the same,
  true system as their template's surfaces, so G13 comes first (fsaverage in `freesurfer-tkr`, m).

**The computation** (a pure function in `compute/`, e.g. `nxr::montage::tenFive(scalp vertices, faces, Nz, Iz, LPA, RPA)`):
- the midline arc Nz → Cz → Iz and the coronal arc LPA → Cz → RPA, each the intersection of the scalp mesh with the plane through its three points;
- Cz found where the two arcs cross at their 50% points (iterated);
- the remaining contours at the rule's percentages along mesh-plane intersections, as Oostenveld and Praamstra (2001) describe;
- on a sphere every arc is a great circle and the result must equal `spherical_1005`.

**The second named option** (`placement: sphere_projection`):
- fit a sphere to the scalp;
- place `spherical_1005` by the landmarks;
- project each point radially onto the mesh, as MNE does for its sphere montages and Brainstorm's `channel_project` does for its caps.

It is simpler, and it gives different positions on a non-spherical head. That is why the default is the arc rule.

**Stored in the template package:**
- the manifold `eeg_1005`, a points manifold, one point per label, with its geometry in the template's head coordinate system (`scs` for
  ICBM152, `freesurfer-tkr` for fsaverage after G13), in metres;
- a label dictionary, with the old-name aliases;
- the `system` partition (10-05, 10-10, 10-20);
- the landmarks used, as `eeg_1005_headpoints` (the D166 shape: points with kind and label partitions);
- the scalp surface it was computed on, as a manifold;
- an **origin record** on each item:
  - source: "computed by nxr-neuro, 10-05 rule (Oostenveld and Praamstra 2001), on <scalp> of <template>";
  - licence and redistributable: the scalp's.

  The release then puts it in the redistributable edition exactly when the scalp is there.
- `producer_json`: the rule, the landmarks' coordinates, the placement option and the arc-intersection tolerance.

### 4.4 A recording with no digitised positions

Positions come from a template montage, matched by channel name. This happens in a pure node and a definition, not in the importer.

- **Matching** (pure node `match_channels`: names in → montage labels or none):
  - case-insensitive;
  - strip a modality prefix (`EEG `) and reference suffixes (`-REF`, `-Ref`, `-M1`, `-A1`, `-LE`);
  - the old-name aliases;
  - the normalisation rules are named options, recorded.

  MNE's `set_montage(match_case=False, match_alias=…)` is the reference for which names match.
- **Placement** (definition `place_sensors`: the recording's sensor manifold, plus a template montage found by its reference on the template
  search path):
  - writes a SIBLING geometry of the sensor manifold (D26: coordinates per vertex that share the topology) in the template's coordinate system,
    anchored into the subject through `register_to_template`'s anchor rows;
  - never overwrites digitised positions;
  - unmatched channels get no position, and are listed.
- **The record** (the geometry's `producer_json`):
  - `placement`: the montage (template name, edition, version, content hash, manifold);
  - the matching rules;
  - `matched` {channel: label}, `unmatched` [channels] and `aliases_used`;
  - `warnings`: "template positions, not digitised: the head's shape and the cap's placement on it are unknown".
  - This is the format-mapping audit's Q3: without fiducials the positions are comparable to the template's anatomy only.
- **A subject with its own MRI and fiducials:** carrying the template montage to the subject's scalp (an affine from NAS, LPA and RPA, then
  projection) is a later option. It is not in the first PRs.

---

## 5. Ordered PR plan, with goldens

Goldens are synthetic only. A MATLAB generator only CALLS Rheome (`gen_*.m`, R2026a, one batch run at a time). A Python generator only calls
MNE-Python, scipy or `eeg_positions`. Each golden sits next to its generator in `*/test/golden/`.

**Sleep events**

0. **Rheome, before the nxr-neuro side** (Q2): expose the slow-wave detection as a public function, `rheome.detect.slowwave`.
   - This is `measure_slowosc.m`'s preparation (lines 156–159) and `i_detect` (321–345), moved unchanged, with `measure_slowosc` calling it.
   - Without it, a generator can reach the detector only by running all of `measure_slowosc` on a synthetic study cache, as
     `tests/tScaleSlowOsc.m` does.
1. **Compute blocks** (`compute/`, unit tests):
   - Butterworth band-pass of order N, zero phase, as a named option of `nxr::filter`, with MATLAB `filtfilt`'s ends;
   - the moving mean (`movmean` and `boxsmooth` conventions);
   - dyadic decimation;
   - `slowWaveCandidates`;
   - `clusterInstants`;
   - `peakFrequency`;
   - an exact streaming median.

   Goldens: scipy (`butter`, `filtfilt`, `hilbert`, `downsample`) for the filters. The detection blocks are checked through the end-to-end goldens
   of PRs 2 and 4.
2. **`sleep_spindles`, per-channel detection** (nodes `downsample`, `smooth`, sample-level `detect_spans`; the `band_filter` IIR option).
   - Golden `gen_spindle_golden.m` calls `rheome.detect.spindle` on synthetic multichannel signals at fs = 800 Hz, so that work rate 100 Hz is
     fs / 8 and dyadic.
   - The signals plant 11–16 Hz bursts in noise, including:
     - one sparse channel;
     - durations at 0.45, 0.5, 3.0 and 3.1 s;
     - gaps of 0.09 and 0.11 s;
     - a burst that reaches only `loSD`;
     - a flat channel (robust SD 0);
     - a baseline mask.
   - The same signals are also run with `workRate = fs` (no resampling), to isolate the resampler.
   - Parity: the event count exact; start and end within one work-rate sample (the resamplers differ), exact with no resampling; `peakFreq`
     within one bin; `peakAmp` to 1e-2 relative. The result must not depend on the chunk size.
3. **Global spindles and the per-event measures:**
   - `join_rows`, `events: per_channel`;
   - `measure` over a spans Selection (`duration`, `amplitude`, `envelope_max`, `peak_time`, `peak_frequency`, `interval_next`), per
     (event × channel);
   - `summarize`;
   - optionally `risefall`.

   Golden: PR 2's Rheome tables (`peakSec`, `peakFreq`, `peakAmp`, `durationSec`) against the measurements. The union is tested on hand-made
   overlaps.
4. **`slow_waves`** (the reference step, `slowWaveCandidates`, `clusterInstants`, the measures).
   - Golden `gen_slowwave_golden.m` calls `rheome.detect.slowwave`, or the PR 0 fallback, on synthetic multichannel signals with planted negative
     half-waves at known lags.
   - The cases straddle every criterion:
     - troughs at −79 and −81 µV;
     - half-wave lengths at 0.29, 0.31, 0.99 and 1.01 s;
     - peak-to-peak at 139 and 141 µV;
     - 4 and 5 channels;
     - two clusters 0.19 and 0.21 s apart;
     - a group that falls short, consuming its candidates;
     - a half-wave across a chunk edge;
     - a record that ends inside a half-wave.
   - Parity: exact events, times and seed channel; trough and peak-to-peak to 1e-9 V after the filter golden.
5. **Docs:** the methods' sections in `docs/reference.md`: the why, the default and its reference with the date, what is stored, and what it
   cannot resolve (section 1.1's and 2.1's limits).
6. **Later, with a spec of its own:** the cortical WHERE per event (section 3), and slow-wave travel (`onsetspread`, sensor latency) with
   Rheome's lagged-generator null.

**Montages**

- M0. **Template inputs:**
  - the scalp (`tess_head.mat`; fsaverage's `outer_skin.surf`) and the landmarks in the importers;
  - their lines in `template_sources.json` with licences and checksums;
  - G13 (fsaverage in `freesurfer-tkr`, m).
- M1. **The 10-05 rule** (`nxr::montage`). Golden `gen_montage_golden.py` calls MNE `make_standard_montage('spherical_1005')` (and so
  `eeg_positions`) and writes the labels and positions.
  - On a synthetic unit sphere, every position must match to 1e-6.
  - On a synthetic ellipsoid, the arc fractions must be exact (checked inside the test; there is no external reference).
  - `sphere_projection` on a sphere must equal the rule.
- M2. **The montage in the template packages:**
  - `create template --montage 1005`;
  - the `system` partition;
  - the origin records;
  - the redistributable edition.

  The by-hand check against MNE's `fsaverage_1005`, `_1010` and `_1020` on the real fsaverage (pointed to by an environment variable, skipped
  otherwise) reports the distance distribution. Nothing measured is committed.
- M3. **`match_channels` and `place_sensors`:**
  - Golden `gen_montage_match_golden.py` calls MNE's `set_montage` matching on synthetic channel-name lists (cases, prefixes, suffixes, the old
    names, unknowns). The matched and unmatched sets must be identical.
  - The sibling geometry and its record are checked in a pipeline test on a synthetic recording and template.

---

## 6. Open questions, each with a recommendation

- **Q1. The spindle reference.** `detect.spindle` cites no paper.
  - **Recommendation:** name Rheome's function as the reference, with its date, in `docs/reference.md`.
  - Add a literature detector, or nxr's `sleep_events` rule (4 × / 6 × the median over channels at NREM samples), only as a named option, and
    only if asked.
- **Q2. Rheome's slow-wave detector is a local function.**
  - **Recommendation:** a Rheome PR that exposes it unchanged (`rheome.detect.slowwave`) before PR 4, so the generator only calls Rheome.
  - The fallback, running `measure_slowosc` on a synthetic study cache, works but is slow and couples the golden to the research measurement.
- **Q3. Spindles: per-channel reference against global events.**
  - **Recommendation:** detect per channel as Rheome does (the golden is tested on that intermediate), then store GLOBAL events: the union across
    rows, `min_channels` 1, the channels recorded, every channel measured within every event.
  - Per-channel storage is the option `events: per_channel`.
  - Because the reference's output is per channel, the maintainers confirm that the default stores the join.
- **Q4. The work rate.** Rheome resamples to exactly 100 Hz.
  - **Recommendation:** downsample by 2^k to the slowest rate at or above 100 Hz (1200 → 150, 1000 → 125, 250 → 125, 200 → 100), so the work-rate
    samples nest in the recording-anchored dyadic tiles (the time-tile rule).
  - Durations are converted to samples at that rate.
  - The goldens run at fs = 800 Hz, where 100 Hz is dyadic.
  - The rate used is recorded.
- **Q5. The spindle band against D84.**
  - **Recommendation:** detection on samples has its band as a filter parameter. D84 governs tile measurements over constant-Q levels, not
    sample detectors. Record this in the spec.
  - The per-event measurements are over the events Selection, so no level is needed.
- **Q6. The median and MAD over hours of samples.**
  - A 100 Hz trace per channel for a night is millions of samples a row.
  - **Recommendation:** an exact two-pass streaming selection (a counting histogram, then the bracketing bin's values), stored as the `baseline`
    statistic with the span.
  - Alternatively, store the work-rate trace as a Field: that item is already listed as not built, as "the trace stored as a Field, the decimated
    trace".
  - Not a percentile approximation: the reference is exact.
- **Q7. Where the per-event measures are computed.**
  - **Recommendation:** a second pass, `measure` over the events Selection (route b in section 1.3). It generalises to imported events and
    holds no signal in memory.
  - Check that the runner allows a definition two passes over the recording, or make it two definitions that share the events by name.
- **Q8. Slow-wave thresholds in absolute volts.**
  - **Recommendation:** the parameters in V (−80e-6 V, 140e-6 V) on EEG channels (the `type` Selection) by default, with the reference
    preparation explicit and recorded.
  - Refuse nothing (tools, not instrument limits).
  - Record in the method's section that the counts depend on the reference.
  - A relative threshold (nxr's 20th percentile of N3 troughs) is a later named option and needs imported stages.
- **Q9. A slow-wave event's extent.**
  - **Recommendation:** a span over the union of its negative half-waves, with `trough_time` as a measurement.
  - `extent: instant` (start = stop at the median trough) as Rheome's form.
- **Q10. Montage landmarks.**
  - **Recommendation:** Nz, Iz, LPA and RPA as the rule's landmarks.
    - Nz, LPA and RPA come from each template's stored fiducials.
    - Iz is defined once per template, as the scalp's posterior-midline point at the inion's standard MNI position, recorded in
      `producer_json`.
  - `eeg_positions` uses T9 and T10 rather than the preauricular points for the coronal arc on a sphere. On a sphere they coincide, but not on a
    head, so the choice is recorded.
- **Q11. Vendor caps.**
  - **Recommendation:** none shipped until each file's terms are confirmed and listed in the manifest. Users bring their own through the readers.
  - Brainstorm's ICBM152 caps are never committed. They are used only as a by-hand check.
- **Q12. fsaverage's scalp.**
  - **Recommendation:** read MNE's fsaverage `bem/outer_skin.surf` (FreeSurfer surface format, read natively), with its origin recorded as
    derived from FreeSurfer's fsaverage under FreeSurfer's terms.
  - If its terms cannot be confirmed, ship fsaverage's montage only in the full edition.
- **Q13. Names.**
  - **Recommendation:**
    - the types `spindle` and `slow_wave`;
    - the definitions `sleep_spindles` and `slow_waves`;
    - the montage manifold `eeg_1005` with `system` members `10-05`, `10-10` and `10-20`;
    - new measure codes from the reserved range (1127 onward), named descriptively (`peak_frequency`, `peak_time`, `trough_time`,
      `peak_to_peak`, `n_channels`, `seed_channel`).
