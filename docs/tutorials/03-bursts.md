# Tutorial 3: bursts of a band

**What this answers.** Rhythms are not steady: activity in a band comes in bursts, and a travelling or rotating pattern exists during a burst, not between them. To measure a
pattern you first need to know *when* the band is active. nxr-neuro labels the bursts of a band on the recording's time axis, with their amplitude, duration and spacing, and
it states the time grid it detected on, so what it can and cannot resolve in time is part of the result. You will detect bursts in the synthetic recording of tutorial 1, whose
10 Hz source was planted in eight bursts at known times, and check the detection against them.

Everything below was run with the toolkit's `nxr-neuro` command on the synthetic dataset (`work/study`, subject `sub-01`, recording `task-rest`; see
[tutorial 1](01-psd.md)). Outputs are excerpts.

## 1. The band's time grid

A band is stated as a quantity in hertz, and the time resolution follows from it: bursts are detected over tiles of the recording's time line whose length is set by the band
(the first power-of-two number of samples that holds four cycles of the band's centre). Make that grid once per band:

    nxr-neuro run select_time_grid --dataset study --subject sub-01 --recording task-rest --rule band --band "8-16 Hz" --all

    {"stage": "result", "pipeline": "select_time_grid@1", ..., "stored": [{"output": "grid", "kind": "selection", "name": "task-rest time grid band 8-16 Hz", ...}]}

Skipping this step is refused with the command to run first: the error says `nothing in the subject matches ... the band's time grid (nxr-neuro run select_time_grid --rule band --band ... first)`.

## 2. Detect the bursts

    nxr-neuro run band_bursts --dataset study --subject sub-01 --recording task-rest --all --combine mean

`--combine mean` detects on one trace, the mean amplitude over channels. The default, `each`, detects per channel; on this 32-channel recording it stores 300 bursts, one set per
channel. The band defaults to 8-16 Hz; a burst is entered at 1.5 robust standard deviations above the median amplitude and extended while above 1.0, gaps up to 0.1 s are joined and bursts
shorter than 0.4 s are dropped (`--enter-sd`, `--exit-sd`, `--merge-s`, `--min-s`; all in the [band_bursts page](../pipelines/band_bursts.md)).

    {"stage": "run", "pipeline": "band_bursts@2", ...}
    {"stage": "result", "pipeline": "band_bursts@2", ..., "hit": false, "run": {..., "halo": 41, ...},
       "stored": [{"output": "bursts", "selection": "band_bursts.bursts.task-rest", ..., "spans": 9},
                  {"output": "baseline", "kind": "statistic", ...}, {"output": "burst_stats", "kind": "statistic", ...},
                  {"output": "tiles", "kind": "measurement", ...}, {"output": "burst_values", "kind": "measurement", ...}]}

Nine bursts were found. The `halo` of 41 samples is the context the band filter needs beyond a chunk's edges, read with the chunk so that the edge does not distort the amplitude.

## 3. Check against what was planted

The bursts are a **spans selection** on the time line: `band_bursts.bursts.task-rest`. Each stored package is a SQLite file, so for a check you can read the spans straight from its
`selection_span` table (the format is described in `io/schema/SPEC.md`); in Python, reading it is `nn.read("study", "sub-01", "band_bursts.bursts.task-rest")`, not run here.

```python
import sqlite3, json
c = sqlite3.connect("file:study/sub-01.nxr.zarr/subject.sqlite?mode=ro", uri=True)
sid = c.execute("select id from selection where name='band_bursts.bursts.task-rest'").fetchone()[0]
found = [(round(a, 2), round(b, 2)) for a, b in c.execute("select start, stop from selection_span where selection_id=? order by ordinal", (sid,))]
planted = json.load(open("data/truth.json"))["bursts_10hz"]
```

    found   = [(4.1, 5.63), (11.26, 13.31), (18.94, 20.48), (26.62, 28.67), (33.28, 35.33), (40.96, 42.5), (47.62, 49.66), (53.76, 55.81), (59.9, 60.42)]
    planted = [[4.0, 5.5], [11.5, 13.0], [19.0, 20.5], [27.0, 28.5], [33.5, 35.0], [41.0, 42.5], [48.0, 49.5], [54.0, 55.5]]

All eight planted bursts overlap a detected burst. Read the differences as the instrument's resolution, not as errors to remove:

- Detected starts and stops lie on the time grid, whose tile is 128 samples (0.512 s) for this band (four cycles of its 12 Hz centre need 83 samples; the next power of two is 128): 4.096 = 8 x 0.512, 11.264 = 22 x 0.512. A burst's time cannot be stated finer than the tile.
- The planted bursts are Hann-shaped and 1.5 s long; the detection extends a tile or so beyond them because the amplitude of a tile averages over its whole length.
- The ninth burst, 59.9 to 60.4 s, is the last tile of the recording, cut at its end: an edge effect, with no planted burst under it. A recording's edge is where a band's amplitude cannot be
  measured properly, and the result carries it as a short burst rather than hiding it.

## 4. The bursts' statistics

Each burst has measurements (amplitude, peak, duration, interval to the next burst), and across the bursts the pipeline stores their statistics: count, sum, median, mean, sd and the 25th, 75th
and 90th percentiles. Export them:

    nxr-neuro export tsv --package study/sub-01.nxr.zarr --statistic band_bursts.burst_stats.task-rest --out out/burst_stats

    {"stage": "result", "files": ["out/burst_stats.tsv"], "sidecar": "out/burst_stats.json", "rows": 32, "columns": ["row", "measure", "summary", "value"]}

The table's `measure` index follows the pipeline's `measures` list (0 amplitude, 1 peak envelope, 2 duration, 3 interval to the next burst) and `summary` follows `burst_summaries`
(0 count, 1 sum, 2 median, 3 mean, 4 sd, 5 25th percentile, 6 75th, 7 90th). For duration (measure 2) in this run:

    row  measure  summary  value
    0    2        0        9                    # count
    0    2        2        2.048                # median duration, seconds
    0    2        6        2.048                # 75th percentile

and interval to the next burst (measure 3) has a count of 8, one fewer than the bursts, since the last burst has no next one, and a median of 5.376 s (a mean of 5.12 s).

## What you learned and where next

- A band is a quantity; the time grid it implies is made, named and stored, and the detection is over that grid.
- Detection thresholds are stated in robust standard deviations and in seconds, and every one is an option recorded with the result.
- The grid's tile is the instrument's time resolution; the edge of a recording is where it fails first.

The method, its reference and what it cannot resolve: [reference, events on the time axis](../reference.md#events-on-the-time-axis). Where the bursts occur in space and how
patterns move inside them is the next layer, [global_bursts](../pipelines/global_bursts.md) and [patches](../pipelines/patches.md). Back: [tutorial 2](02-source-map.md).
