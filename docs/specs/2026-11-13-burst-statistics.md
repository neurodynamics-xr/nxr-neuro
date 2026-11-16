# Burst statistics: measurements of each event, statistics over the events (spec, 2026-11-13; APPROVED by the maintainers;
# the build's decisions in *How it is built*, below)

## Why

`band_bursts` labels the time axis with bursts, but stores nothing ABOUT them: no burst's amplitude or duration, and no count, rate or share of time.
Electrophysiology describes transient rhythms with a standard set of measures. Two levels:
- **Per event:** onset, offset, duration (in s and in cycles), peak and mean amplitude, the rise and the fall.
- **Over the events:** count, rate per minute, the share of time in bursts, and the distributions of duration, amplitude and inter-burst interval.

They follow the layers exactly:
- per-burst values are MEASUREMENTS over the bursts' Selection (each burst a member);
- the summaries across the bursts are STATISTICS of those measurements: the statistic summaries (method `summary`, step `summarize`,
  `docs/specs/2026-11-13-statistic-summaries.md`), computed across the members, never within one.

The bursts are a partition of the time axis exactly as tiles are: measured within each member, summarised across the members.

## References (to be audited, read and checked in code, before each is implemented; nothing ported unread)

- **Rheome `detect.eventtable` / `detect.risefall`:**
  - per event: tOn and tOff (the half-rise and half-fall times of a double-sigmoid fit), `durS`, build-up 4τR, plateau, decay 4τF, `height`, `peakAmp`
    (the envelope's max in [tOn, tOff]), r²;
  - an unfitted event is counted, never hidden.
- **Rheome `detect.occupancy`, nxr's burst summary** (`n_bursts`, `burst_cycles`, `burst_share`): the share of time a state is on.
- **The literature's standard descriptors of transient bursts:**
  - event rate, duration, peak power and frequency span (Shin et al. 2017, eLife: beta events);
  - duration and amplitude distributions, and their relation (Tinkhauser et al. 2017, Brain);
  - cycle-by-cycle burst features: number of cycles, period, amplitude, rise-decay symmetry (Cole & Voytek 2019, bycycle).
  - These are to be checked against the papers and the bycycle code before any of them is adopted.

## Per-burst measurements (the measurement layer)

**Stored as the windowed amplitude is:** array-backed, the member axis the bursts in the Selection's order.
- The array is [channels × bursts × components]. A burst's value sits in the row of the channel that found it, NaN in every other row (`combine mean`:
  one row).
- Reason for this form: in an event set the code is the event TYPE, so scalar rows keyed by code cannot tell bursts apart. The array's member axis can
  (the audit, `docs/audits/2026-11-13-time-event-labelling.md`).
- The rows place each burst in its channel. The statistic summaries per row then give per-channel burst statistics with no extra grouping.

| measure | per burst | kind |
|---|---|---|
| `duration` | stop − start, s | geometric: the span itself (the time analogue of a parcel's `area`) |
| `n_cycles` | duration × the band's centre frequency | geometric, given the band |
| `amplitude` | the mean of the band's amplitude \|z\| over the burst's good samples | the measurement function `amplitude` over the burst |
| `max` (peak) | the max of \|z\| over the burst; `peak_time` its time | the measurement function `max` over the burst |
| `relative_peak` | peak / the baseline's median (derived at read) | a ratio, derived |
| `ibi_next` | the time from its end to the next burst's start on the same channel (NaN for the last) | geometric, over the event set |

- **Superseded:** a detected burst is measured from the tile measurements it covers, by their exact roll-up, not by another pass over
  the samples; see *Tiles and events* and *How it is built* below, which say which of these measures are built now.
- **Later, separately:** the rise and fall fit (Rheome `risefall`: tOn, tOff, τR, τF, plateau, r²), and the burst's peak frequency (it needs a spectrum
  per burst).

## Burst statistics (the statistic layer: across the bursts)

The statistic summaries across the members of the bursts Selection, per channel (`each`), pooled, or for the mean trace:
- **count:** the bursts (the finite values per row; a new summary `count`);
- **rate:** count / the span's good time, per minute. The span and its good time are recorded with the statistic, so the rate is derived at read
  (a ratio is never stored as a reduction);
- **occupancy:** Σ duration / the good time (the share of time in bursts; nxr's `burst_share`, Rheome's occupancy). A new summary `sum`; the ratio is
  derived at read;
- **duration, amplitude, peak, n_cycles:** median, mean, sd and percentiles (25, 75, 90): the distributions the literature reports;
- **inter-burst interval:** median, mean, and its coefficient of variation sd / mean (derived at read).

They are stored by `band_bursts` itself (below, *How it is built*), read if stored.

## Tiles and events: two kinds of Selection

- **Tiles** are a dense, regular PARTITION of the whole time axis (each constant-Q level one band). Their measurements are the windowed features of the
  recording (as MATLAB's frames): the band's mean amplitude and its maximum per tile. They are not events.
- **Events** are a SPARSE Selection a detector derives from those tile measurements: a burst is not a member of any tile level; it is a run of tiles
  of its band's level (a 10 s burst covers several 2 s tiles).

## How it is built

1. **`band_bursts` stores one more windowed feature:** the band's maximum per tile (`envelope_max`, the Studio's code 25), beside `amplitude`, over the
   same level, in the same chunk loop.
2. **A burst is measured from the tile measurements it covers, exactly, with no second pass over the recording.**
   - A detected burst starts and stops on tile edges, so it IS the union of its tiles; its measures are the roll-up of theirs (the registry's
     reductions, as the lookup merges members):

   | measure | per burst | from its tiles |
   |---|---|---|
   | `amplitude` | mean \|z\| over the burst | the count-weighted mean of the tile means (reduction mean) |
   | `envelope_max` | its peak \|z\| | the max of the tile maxima (reduction max) |
   | `duration` | stop − start, s | geometric: its tiles × the tile width (new, reduction sum) |
   | `interval_next` | the next burst's start on a shared row − this stop, s; NaN for the last | geometric (new, reduction none) |

   - **The band** is the level's, the one the bursts were detected from (their `params_json` names the measurement and the level): no ambiguity, and
     nothing added to a row's key.
   - **A member is its tiles × its channels:** a burst found on a channel (`params_json.channels`) is measured on that row, NaN on the others; a burst
     of the mean trace (`combine mean`) on every row.
   - A burst holding a tile that is not whole and good is NaN for the sampled measures (the tiles' rule); its geometry is still its own.
   - **Not here:** a median per burst (medians do not roll up; it needs the samples).
   - Stored array-backed over the bursts Selection, [rows × bursts × 2] (the value, the samples counted), the member axis the bursts' order.
3. **The statistics across the bursts** (`summarize`, one statistic per measure): new summaries `count` (the finite values) and `sum`, with `median`,
   `mean`, `sd`, `percentile:25/75/90` by default. Each records its span and the span's good seconds (`good_s`), so the rate (count / good_s × 60),
   the occupancy (sum of duration / good_s) and the CV (sd / mean) are derived at read.
4. **Later, separately:**
   - events NOT derived from tiles (imported Brainstorm events: arbitrary edges) are measured from the samples, a generic `event_measures` pipeline;
     the band then needs a place in the row's key (open);
   - `n_cycles` (waits for the per-burst peak frequency), `relative_peak` (a ratio across a measurement and a statistic), Rheome's rise and fall fit.

## Tests

- **Planted bursts of known durations, amplitudes and gaps on synthetic channels:** each burst's measurements equal the planted ones to a tile; the
  count, rate, occupancy and interval statistics equal direct computations.
- **One burst per channel versus several:** the per-channel rows.
- **`combine mean`.**
- **A rerun is a hit.**

## Checklist (the architecture policy)

- **Compute:** pure functions of the band's amplitude per chunk, the spans and the measurement values (`nxr::intervals`, `nxr::stats`).
- **io:** reads the stored rows and writes the measurement and the statistic.
- **Orchestration:** names the event set, the measurement and the summaries. Ratios (rate, occupancy, relative peak, CV) are derived at read.
