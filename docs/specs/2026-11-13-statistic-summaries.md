# Statistic summaries: statistics of a measurement across its partition's members (spec, 2026-11-13; APPROVED by the maintainers;
# amended 2026-11-13, maintainers:: renamed from "scalarization", the MATLAB comparison corrected)

## Why

A measurement function reduces the ELEMENTS of a partition member to one value: the mean of a tile's samples, the max of a parcel's vertices (the
Studio's `measureOverPartition`). The layer above needs the same kind of step one level up: reduce the MEMBERS of a partition to a statistic. Examples:
- the median and robust spread of a channel's alpha amplitude over all of a recording's tiles;
- the 75th percentile of the tile power.

Today every detector computes its own baseline inside its step and keeps the numbers only as text: `band_bursts` computes median and MAD in
`detect_spans`.

**How this compares with MATLAB's signal feature extractors** (`signalTimeFeatureExtractor`, `signalTimeFrequencyFeatureExtractor`; corrected
2026-11-13). This step does NOT correspond to MATLAB's `ScalarizationMethod`; the first draft said it did, and named the step after it.
- MATLAB cuts the signal into frames (`FrameSize`, `FrameOverlapLength`) and computes each feature per frame: a matrix of frames × features.
- `ScalarizationMethod` (mean, standard deviation, median, peak, kurtosis, …) works WITHIN a frame: it reduces a feature with several values in one
  frame (e.g. a time-frequency feature, one value per time instant of the frame) to one number for that frame.
- Statistics across the frames are left to the user or a classifier.

| MATLAB | nxr-neuro |
|---|---|
| the signal | a Field |
| frames | a partition Selection (dyadic tiles, chunks, events, parcels) |
| a feature per frame, `ScalarizationMethod` within it | a measurement function per member: the reduction within a member (mean, rms, peak, `mean_abs`, the factors below) |
| nothing (left to the user) | this step: a STATISTIC across the members |

The same arithmetic can sit in either layer: within a member it is a measurement function, across members a statistic summary. Within-member
reductions MATLAB has and we lacked:
- **skewness and kurtosis** (added 2026-11-13, maintainers): two stored raw moments, `mean_cube` (Σw·x³/Σw) and
  `mean_fourth` (Σw·x⁴/Σw), accumulated and rolled up as the mean is; `skewness` = m3/m2^1.5 and `kurtosis` = m4/m2² (1/n, MATLAB's, not excess) derived
  at read from the rolled-up mean, mean_square and those (a ratio of sums, never a sum of ratios). Limit: raw moments cancel when the mean is large
  against the spread.
- **the median**: not sums-based; built exactly (below, *The median within a member*).

### The median within a member

Two things make it unlike every other measurement function: it cannot be accumulated from sums (a member spanning chunks needs all its samples at
once), and it does not roll up (the median of two members together is no combination of their medians). Options B (a histogram per member: binned,
the bins the caller's) and C (no median within a member) were not taken.

**Built exactly:**
- **compute:** `nxr::partition::medians(codes, n, members, values)`: per member, the median of its finite values (numpy's: the mean of the two middle
  ones for an even count), NaN for an empty member. Unweighted only: a weighted median is not built, so `measure compute` refuses `median` with a weight
  other than `none`, with the reason.
- **time members** (the runner's `measure`: the run's span, the tiles of a level, an event set's members): each member's good samples held per row until
  the member closes, then reduced. Memory: the longest member's samples × rows. A member longer than 2^20 samples (about 15 minutes at 1200 Hz) is
  refused, never truncated. The member rule is unchanged: a member not whole and good in the span is NaN.
- **the registry:** `median` (nxr-neuro's range) with reduction `none`, a new reduction meaning "does not roll up". The lookup returns a member's own
  row and REFUSES to merge rows of a `none` measure (a shift that groups members), with the reason.

## What it computes

**In:** a stored measurement:
- a Selection that partitions something (the tiles of a constant-Q level, a parcellation);
- what was measured (a Field, or a band of a map statistic);
- the measure (`amplitude`, `mean`, `mean_square`, …).

Its values per row (a channel, a vertex) per member (a tile, a parcel) come from either form the model stores:
- array-backed rows ([rows × members × components], as `band_bursts` and `measure over time` store);
- scalar rows (one value per member, as `measure compute` stores).

**The summaries,** over the members whose value is finite (bad tiles, empty parcels left out; their count recorded), per row by default or pooled over
rows (`--pool`):

| summary | definition | reference |
|---|---|---|
| `mean` | Σx / n | MATLAB Mean |
| `sd` | √(Σ(x − mean)² / (n − 1)) | MATLAB StandardDeviation (N − 1); `ddof` an option |
| `median` | the middle value (the mean of the two middle ones for even n) | numpy `median` |
| `mad` | 1.4826 · median \|x − median\| (the robust SD) | Rheome `detect.spindle`, nxr `spindle_envelope` |
| `percentile:p` | linear interpolation between order statistics | numpy `percentile` (linear), the Studio's `percentileLinear` |
| `peak` | max \|x\| | MATLAB PeakValue |
| `kurtosis`, `skewness` | the standardized 4th and 3rd moments (1/n) | MATLAB Kurtosis, Skewness |
| `min`, `max` | | |

**Out:** a STATISTIC node:
- measure = the measurement's measure (e.g. `amplitude`);
- method `summary`;
- value_type scalar;
- OF what was measured (`of_field_id`, or `of_statistic_id` for a band of a map statistic);
- partition = the Selection, n_members = the members summarised (`members_skipped` = those left out);
- axes = row (the rows' manifold or Selection) × summary (each entry named, with its parameters: `{name: "percentile", p: 75}`);
- span = the measurement's;
- options = the summaries, `pool` and `ddof`;
- the cache key = the measurement's identity (Selection, what was measured, the measure, the rows' `computed_by`) and the options.

A rerun with the same key is a hit. Read if stored, computed and stored if not (decision 1 of the event spec).

## Where it runs

- **A runner step `summarize`** (after: in a measurement of the run, out a statistic) and a runner output of kind `statistic`. A definition summarises
  what it measured in the same run and stores it.
- **A verb `statistic compute`** for a measurement already stored: `--labelling`/`--over`, `--of`, `--measure`, `--summaries`. It reads the rows through
  io and computes, with no recording read.
- **`detect_spans`** gains an optional input `baseline` (a `summary` statistic with `median` and `mad`). `band_bursts` then:
  - measures the amplitude;
  - summarizes it (median, mad per channel);
  - detects with the stored baseline;
  - stores the statistic beside the bursts.

  Its bursts are unchanged (tested: identical spans). Their `params_json` names the baseline statistic instead of copying its numbers.

## The measure registry: MATLAB's time-domain features as measurement functions

Two new STORED measures (nxr-neuro's range, `measures_nxr.json`), accumulated with the others by the measurement functions:
- `mean_abs` (mean |x|);
- `mean_sqrt_abs` (mean √|x|).

Three DERIVED at read from the rolled-up ingredients (a ratio of sums, never a sum of ratios), as `sd` and `crest_factor` are:

| measure | from | MATLAB |
|---|---|---|
| `shape_factor` | rms / mean_abs | ShapeFactor |
| `impulse_factor` | abs_max / mean_abs | ImpulseFactor |
| `clearance_factor` | abs_max / mean_sqrt_abs² | ClearanceFactor |

`crest_factor` (abs_max / rms) and `rms` already exist. Together they are MATLAB's time-domain feature list, except SNR, THD and SINAD, which need a
spectrum. The Studio's `reduceTiles` (a ported reference) is not changed: the new sums are nxr-neuro's own accumulation, beside it.

## Not in this spec

- **Time-frequency features per tile** (spectral entropy, flatness, kurtosis, crest; instantaneous frequency and bandwidth): a later family, if wanted.
- **Overlapping feature frames** (MATLAB's `FrameOverlapLength`): the time partition stays the dyadic tiles.

## Tests (scoped)

- **The summaries against direct computations** on synthetic values: the same numbers as numpy for median and percentile, and as MATLAB's definitions
  for sd, kurtosis and skewness (a small golden from numpy/scipy).
- **Statistics of both forms:** array-backed rows and scalar rows, per row and pooled; members with NaN left out and counted.
- **`band_bursts` with its baseline as a statistic:** the same spans as before; the statistic stored; a rerun a hit.
- **The new measures** against direct computations over a partition.

## Checklist (the architecture policy)

- **Compute:** the summaries are pure functions of a vector (`nxr::stats` or the existing `nxr::intervals::median`/`mad` generalised).
- **io:** owns reading the measurement rows and writing the statistic.
- **Orchestration:** names the measurement and the summaries. Every choice is the caller's and recorded.
