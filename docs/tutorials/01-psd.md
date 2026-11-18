# Tutorial 1: the power spectrum of a recording

**What this answers.** Before any pattern is measured, you need to know which rhythms a recording holds and at what frequency resolution you can tell them apart. A spectrum
states both: power per hertz, and the width of a frequency bin. You will make a small synthetic EEG dataset with known rhythms, import it, compute its Welch power spectrum with
nxr-neuro, read it as a table and check the peaks against what was put in. Everything below was run with the toolkit's `nxr-neuro` command (version 0.1.0); the outputs shown are excerpts.

Synthetic data only: the dataset is made by [synthetic_bids.py](synthetic_bids.py) (NumPy only). It writes 32 EEG channels at 250 Hz for 60 s on a spherical head: a 10 Hz source that
comes in eight bursts (used in tutorial 3), a weaker steady 20 Hz source elsewhere, and white noise, as an EDF file with BIDS tables (channels, electrode positions in metres).

## 1. Make the data and import it

    mkdir work && cd work
    python3 ../docs/tutorials/synthetic_bids.py data
    nxr-neuro import bids --from data/bids --dataset study

`import bids` makes the dataset folder `study` with one subject package per BIDS subject and registers it. The last line of its output names what it stored:

    {"stage": "result", "dataset": "study", "name": "synthetic tutorial data", "bids_version": "1.9.0", "licence": "unknown", "subjects": ["sub-01"],
     "recordings": [{"name": "task-rest", ..., "n_channels": 32, "n_samples": 15000, "sfreq": 250.0, "events": 0, ...}]}

Look at what is there:

    nxr-neuro list subjects --dataset study
    nxr-neuro list recordings --dataset study --subject sub-01

    {"stage": "result", "subject": "sub-01", "recordings": [{"name": "task-rest", "kind": "recording", "table": "field"}]}

(Output is JSON lines here because stdout is not a terminal; in a terminal you get a readable summary.)

## 2. Compute the spectrum

    nxr-neuro run psd --dataset study --subject sub-01 --recording task-rest --all

`--all` is the time range: **you always state it** (`--from 0 --to 30` would use the first 30 s). The recording is read one chunk at a time. The run prints a progress line per
chunk and a result line whose end says what was stored:

    {"stage": "run", "pipeline": "psd@1", "span": "all 1 chunks", "halo": 0, "steps": 1}
    {"stage": "chunk", "done": 1, "count": 1, "chunk": 0, "seconds": 0.006, "state": "done"}
    {"stage": "result", "pipeline": "psd@1", "subject": "sub-01", "anchor": "task-rest", "hit": false, ...
       "stored": [{"output": "psd", "kind": "statistic", "name": "task-rest_psd_welch", ...}]}

The result is a **statistic** `task-rest_psd_welch` (power per hertz, per channel) plus the segments it averaged. Its options, meanings and units are in the
[psd page](../pipelines/psd.md). The defaults matter for the instrument: by default a Welch segment is a *tile* of 2^k samples (here 1024 samples, 4.096 s, so the
frequency step is 250/1024 = 0.244 Hz), with 50 % overlap and a Hamming window.

Run it again:

    nxr-neuro run psd --dataset study --subject sub-01 --recording task-rest --all

The result line now has `"hit": true`: same pipeline, options, input and range, so the stored result is found and nothing is computed.

## 3. Read it

    mkdir out
    nxr-neuro export tsv --package study/sub-01.nxr.zarr --statistic task-rest_psd_welch --out out/psd

    {"stage": "result", "files": ["out/psd.tsv"], "sidecar": "out/psd.json", "rows": 16416, "columns": ["channel", "frequency", "value"]}

`out/psd.tsv` has a row per channel and frequency (32 channels x 513 bins); `out/psd.json` says what `value` is. Its description and units, and the scaling it was stored with:

    "value": {"Description": "the power_density task-rest_psd_welch", "Units": "channel unit²/Hz"}
    "scaling": {"stored": "density", "band_width_hz": 0.244140625, "bin_enbw_hz": 0.33295518109784794, ...}

`bin_enbw_hz` is the equivalent noise bandwidth of one bin: the real frequency resolution, wider than the bin step because of the window.

Find the peaks of the channel-averaged spectrum:

```python
import csv, collections
acc = collections.defaultdict(list)
for r in csv.DictReader(open("out/psd.tsv"), delimiter="\t"):
    acc[float(r["frequency"])].append(float(r["value"]))
mean = {f: sum(v) / len(v) for f, v in acc.items()}
print(sorted(mean, key=mean.get, reverse=True)[:3])
```

    [20.01953125, 10.009765625, 19.775390625]

The two planted rhythms, 20 Hz and 10 Hz, are the two largest peaks, each within one bin (0.24 Hz) of the truth; the third value is the neighbouring bin of the 20 Hz line, the
window's leakage. That is the instrument's frequency resolution made visible: a line is spread over about 0.33 Hz here, and two rhythms closer than that could not be told apart
at this segment length.

## 4. Ask for another reference's spectrum

The same pipeline reproduces other tools' spectra with presets (`--set brainstorm`, `--set scipy`, `--set nxr-chain`); a preset only sets parameters, and a name you give wins.

    nxr-neuro run psd --dataset study --subject sub-01 --recording task-rest --all --set scipy --name task-rest_psd_scipy

This stores a second statistic beside the first (a different name; otherwise the run would be refused, or with `--replace` replace the first). The scipy preset uses
`round(4 s x 250 Hz) = 1000` samples, a periodic Hann window, so the step is 0.25 Hz and there are 501 bins, and its peaks land on 20.0 Hz and 10.0 Hz exactly, as
`scipy.signal.welch` would give them. Choosing between the two is a choice about the frequency grid, and it is yours.

## What you learned and where next

- A dataset is a catalogue of subject packages; a pipeline stores its results in the package with their provenance, and the same run again is a cache hit.
- The time range is always stated, and recordings are read chunk by chunk.
- The spectrum's resolution is a property of the segment length and window, recorded with the result.

Next: [tutorial 2](02-source-map.md) takes the same recording towards the cortex. Options of every pipeline: [pipelines](../pipelines/README.md); the method and its limits: [reference](../reference.md#the-welch-psd).
