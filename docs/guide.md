# User's guide

nxr-neuro measures the **spatiotemporal patterns of cortical activity** (travelling, rotating and standing patterns) from MEG and EEG, in physical units, at declared
scales, and it says what the instrument cannot resolve. Two things make that hard to do reliably: the instrument (sensors and source estimation) can manufacture apparent
patterns, and results are hard to compare because scales and units are rarely stated. So nxr-neuro keeps every measurement together with the data it was made from, the exact
options that produced it and the time range it covered, and it refuses to guess the choices that belong to you (which recording, which span, which band).

This guide is the short path: install, what a dataset is, run a pipeline, read the result, in the command line, Python and MATLAB. Then the
[tutorials](tutorials/01-psd.md) do it end to end on synthetic data. The method behind each pipeline is in the [reference](reference.md); every pipeline's options are in the
[pipelines index](pipelines/README.md).

Every command shown on this page was run with the toolkit's own command on synthetic data, unless a line says it was not.

## 1. Install

Take a release tarball for your platform, check it, and put the command on your path (full text, including the macOS Gatekeeper note, in the [README](../README.md#install-a-release)):

    sha256sum -c --ignore-missing SHA256SUMS
    tar -xzf nxr-neuro-<version>-<platform>.tar.gz
    mkdir -p ~/.local/bin && cp nxr-neuro-<version>-<platform>/nxr-neuro ~/.local/bin/
    nxr-neuro --version

Python needs the wheel (`pip install nxr_neuro-<version>-...whl`, NumPy only); MATLAB needs the MEX built from the sources (R2021a or newer; see the
[README](../README.md#python-a-first-slice-the-wheel) and `bindings/mex/matlab/README.md`). To build the command from source, follow the README's Build section.

## 2. The dataset

A **dataset** is a folder, named by its path (`--dataset study`). It is a catalogue of **subject packages**:

    study/
      catalogue.sqlite          which subjects are registered, their participant variables, a content hash of each package
      sub-01.nxr.zarr/          one subject: a self-contained folder you can copy, archive or share on its own
        subject.sqlite            the subject's rows: recordings, surfaces, operators, selections, statistics, measurements, provenance
        ...                       the arrays (Zarr v3 shards) and small files the rows point at

A package is complete by itself: opening it needs no catalogue and no other subject. What it holds, by kind:

| kind | what it is | examples |
|---|---|---|
| recording | a field over channels and time, stored in chunks | `task-rest` |
| surface | a triangle mesh in metres, the cortex or a head surface | `cortex`, `scalp` |
| operator | a linear map between manifolds | a head model's gain, an inverse kernel |
| selection | a labelled part of a manifold: spans of time, tiles of the cortex, events | a recording's bursts, a time grid |
| statistic | a summary computed from a partition: a spectrum, a covariance | `task-rest_psd_welch` |
| measurement | a value per member of a selection, with the ingredients to combine members | a burst's amplitude and duration |

Every stored result records its **provenance** (the pipeline, its version and every option, the inputs by name and id, the time range actually used) and a **cache key**. A second
run with the same pipeline, options, inputs and range finds the stored result and computes nothing.

To get data in:

| you have | command |
|---|---|
| a BIDS dataset (MEG, EEG, iEEG) | `nxr-neuro import bids --from <bids folder> --dataset study` |
| a recording file (FIF, EDF, BDF, BrainVision, EEGLAB, CTF, KIT) | `nxr-neuro create dataset --dataset study`, `nxr-neuro create subject --dataset study --name sub-01`, then `nxr-neuro import recording --from file.edf --package study/sub-01.nxr.zarr` |
| a mesh (OBJ, PLY, STL) | `nxr-neuro add surface cortex.obj --dataset study --subject sub-01 --name cortex` (`--scale` brings the file's units to metres) |
| nothing yet | `nxr-neuro create recording --synthetic --package study/sub-01.nxr.zarr --name rest` writes sines plus noise from a seed, to try the toolkit |
| a FreeSurfer or Brainstorm anatomy, a NIfTI, a GIFTI | `nxr-neuro import --help` lists the importers |

Look inside with `nxr-neuro list subjects|recordings|kernels|results --dataset study [--subject sub-01]` and `nxr-neuro show subject --dataset study --subject sub-01`.

## 3. Run a pipeline

One command, verb first. A pipeline is a chain of nodes (pure functions) defined in a YAML file; `nxr-neuro run <pipeline>` runs it on a subject.

    nxr-neuro run --help                              # the pipelines, by topic
    nxr-neuro run psd --help                          # one pipeline: every option with meaning, default, unit; and an example
    nxr-neuro run psd --dataset study --subject sub-01 --recording rest --all

What to know:

- **You state the time range.** `--from <s> --to <s>` (seconds, `m:ss` or `h:mm:ss`) or `--all`; without one the command refuses. Recordings are read one chunk at a time
  with bounded memory, whatever the range, and the range actually used is stored with the result.
- **Parameters are options.** Each parameter of a pipeline is `--<name> VALUE` (dashes for underscores; the value is read as YAML: `8`, `[15, 29]`). `--set <preset>` applies a
  named group of parameters, for example `--set scipy` or `--set brainstorm` on `psd` reproduces that tool's spectrum; an option given by name wins over a preset.
- **Results are never silently overwritten.** A run whose outputs exist with other options is refused; `--replace` moves the old result aside and writes the new one.
- **A repeated run is a cache hit.** The result line says `"hit": true`.
- **Output.** In a terminal the output is readable (a progress bar on stderr, a result line, what was stored). With `--json`, or when stdout is not a terminal, it is JSON lines,
  one per event; `--quiet` prints only the result line.

The result line of a run lists what was stored (an excerpt, from the first tutorial):

    {"stage": "result", "pipeline": "psd@1", "subject": "sub-01", "anchor": "task-rest", "hit": false, ...,
     "stored": [{"output": "psd", "kind": "statistic", "name": "task-rest_psd_welch", ...}]}

The pipelines are listed with their parameters in the [index](pipelines/README.md); `nxr-neuro list pipelines` lists them too.

## 4. Read the results

| to see | command |
|---|---|
| what a subject holds | `nxr-neuro list results --dataset study --subject sub-01` |
| a statistic as a table | `nxr-neuro export tsv --package study/sub-01.nxr.zarr --statistic NAME --out work/psd` (writes `work/psd.tsv` and a JSON sidecar with units, scaling and provenance) |
| measurements of a selection's members, at a scale you state | `nxr-neuro query measurements --dataset study --subject sub-01 --selection NAME --measure NAME` |
| a cortical map for another tool | `nxr-neuro export gifti ...`; a BIDS derivatives folder: `export bids-derivatives`; MNE source estimates: `export fif` |
| a table across subjects, for statistics | `nxr-neuro export measurements ...` |

## 5. Python

The wheel carries the same C++ library as the command; a result made in Python is the same cached result the command finds. (The wheel was not available to run for this
guide, so this section follows the wheel's own documentation and source, not a run: `bindings/python/README.md`.)

```python
import nxr_neuro as nn

nn.list.pipelines()                       # the pipelines, each parameter with its meaning, default and unit
r = nn.run("psd", "study", "sub-01", recording="task-rest", all=True)    # start=, stop= in seconds, or all=True
r.cache, r.stored                         # "computed" or "hit"; what was stored
psd = r.read()                            # psd.values [channels, frequencies], psd.meta
psd = nn.read("study", "sub-01", "task-rest_psd_welch")

# pure functions on arrays, no dataset
res = nn.compute.signal.psd(x, sfreq=250.0)          # x: [channels, samples]; res.frequencies, res.psd
y = nn.compute.signal.bandpass(x, 8, 12, sfreq=250.0)
rec = nn.io.open_edf("rest.edf"); block = rec.read(start=0, stop=10_000)
```

## 6. MATLAB

The `+nxr` package over one MEX (`nxr_compute`): the same names and options as Python, **indices 1-based**, name-value arguments. Not run for this guide (no MATLAB session was
used); the calls follow the help text of `bindings/mex/matlab/+nxr/run.m` and `read.m`.

```matlab
addpath('bindings/mex/matlab'); addpath('build/bindings/mex')
p = nxr.list.pipelines();                                 % nxr.list.help('psd') prints one
r = nxr.run('psd', 'study', 'sub-01', 'recording', 'task-rest', 'all', true);
r.cache, r.stored
s = nxr.read('study', 'sub-01', 'task-rest_psd_welch');   % s.values [channels x frequencies], s.meta
x = nxr.io.read_edf('rest.edf', all=true);                % the span is always stated
y = nxr.compute.signal.bandpass(x, 8, 12, 250);
```

## 7. Where next

- [Tutorial 1: the power spectrum of a recording](tutorials/01-psd.md)
- [Tutorial 2: from a recording to the cortex](tutorials/02-source-map.md)
- [Tutorial 3: bursts of a band](tutorials/03-bursts.md)
- [Pipelines](pipelines/README.md), [nodes](nodes.md), [methods reference](reference.md), [architecture](../ARCHITECTURE.md)
