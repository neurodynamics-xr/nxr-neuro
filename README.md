# nxr-neuro

A C++ toolkit and command line, `nxr-neuro`, for **measuring the spatiotemporal patterns of cortical activity from MEG and EEG**: travelling, rotating and
standing patterns, in physical units, at declared scales, on each person's own cortex.

## Why

Brain activity is organised in space, time and scale: patterns move across the cortex as waves, rotate around cores and stand in place. These patterns are
proposed to underlie perception, memory, action and cognition, and to change with development, ageing and disease. The field cannot yet measure them
reliably. Patterns often lack physical units and are hard to detect in single events; the measuring instrument (sensors, source estimation) can manufacture
apparent waves; ground truth is rare; scales do not link; and results are hard to compare across people.

nxr-neuro is a measurement toolkit for that problem. It measures size, direction, speed, path, divergence and rotation in physical units at declared scales,
places every measurement on a shared space-time atlas, and stores each result with its full provenance. Every method has a named reference implementation it
reproduces (nxr in Python, the Cortical Flow Studio, Brainstorm, MNE-Python, SciPy) and a test that proves the match on synthetic data. The toolkit says what
the instrument cannot resolve; it does not claim more than it measures.

## What is in it

| Folder | Library | What it does |
|---|---|---|
| `compute/` | `nxr_compute` | the maths on a cortical surface: operators, eigenmodes, filterbanks, flow, events (Eigen, Spectra, geometry-central, pocketfft; no file I/O) |
| `io/` | `nxr_io` | the nxr dataset format (SQLite rows plus Zarr v3 arrays; a subject is a self-contained package, store format 53, a dataset a catalogue of packages, and a template a released, versioned package found on a search path), a chunked recording reader, and a Brainstorm protocol importer |
| `meeg/` | `nxr_meeg` | MEG/EEG source imaging: head models, noise covariance, minimum-norm kernels |
| `pipelines/` | `nxr_pipelines` | the glue: read with `io`, compute with `compute` and `meeg`, write with `io`, cached |
| `cli/` | `nxr-neuro` | the command |
| `bindings/` | | MATLAB MEX, Node and WebAssembly bindings of `compute` |

The architecture is strict and documented in [ARCHITECTURE.md](ARCHITECTURE.md): data loading, preparation and type checks, a graph of pure functions, and
read-back, with reductions done only through partition measurements. Every verb and method is described in [docs/reference.md](docs/reference.md); the
designs behind them are in [docs/specs/](docs/specs/).

## Build

You need CMake 3.20 or newer, Ninja and a C++17 compiler, plus SQLite, zlib and zstd from your system (for example `brew install zstd`). Eigen, Spectra,
c-blosc, HDF5 and matio are fetched into the build directory; geometry-central is a git submodule, so clone with submodules:

    git clone --recurse-submodules <repository URL>
    cd nxr-neuro
    cmake -S . -B build -G Ninja
    cmake --build build
    ctest --test-dir build

The tests use synthetic data only. Tests that need data of your own (a real recording, a Brainstorm protocol, reference checkouts) read it from environment
variables and are skipped, with a reason, when the variables are not set; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Install a release

Each release (https://github.com/neurodynamics-xr/nxr-neuro/releases) has a tarball for Linux (`nxr-neuro-<version>-linux-x86_64.tar.gz`, x86_64, glibc 2.28
or newer; no other library is needed) and for macOS (`nxr-neuro-<version>-macos-arm64.tar.gz`, Apple silicon, macOS 13.3 or newer), and a `SHA256SUMS` file.

    sha256sum -c --ignore-missing SHA256SUMS        # macOS: shasum -a 256 -c --ignore-missing SHA256SUMS
    tar -xzf nxr-neuro-<version>-<platform>.tar.gz
    mkdir -p ~/.local/bin && cp nxr-neuro-<version>-<platform>/nxr-neuro ~/.local/bin/     # and have ~/.local/bin on your PATH
    nxr-neuro --version

The macOS binary is **not signed or notarised yet**, so Gatekeeper may refuse a downloaded copy ("cannot be opened because the developer cannot be
verified"). After checking the SHA-256 sum, remove the quarantine mark once with `xattr -d com.apple.quarantine nxr-neuro`, or allow it in System Settings,
Privacy & Security.

## Python (a first slice: the wheel)

A Python face over the same C++ library as the command: pure functions on NumPy arrays and the readers of the field's file formats. It is a first
slice (the dataset, `compute` verbs and exporters come with the dataset format). Wheels for Linux x86_64 (manylinux_2_28) and macOS arm64 (13.3 or newer)
are built by the `wheels` workflow (Actions, "wheels", on a version tag or by hand) and attached to the run as artefacts; nothing is published to PyPI yet.
One wheel per platform serves every Python 3.12 or newer. Only NumPy is required.

    pip install wheel-linux-x86_64/nxr_neuro-<version>-cp312-abi3-manylinux_2_28_x86_64.whl      # or the macOS wheel

    import nxr_neuro as nn
    r = nn.compute.signal.psd(x, sfreq=1000.0)                 # x: [channels, samples]; r.frequencies, r.psd (scipy.signal.welch's values)
    y = nn.compute.signal.bandpass(x, 8, 13, sfreq=1000.0)
    c = nn.compute.stats.pca(table)                             # observations x features
    rec = nn.io.open_edf("rest.edf")                      # header now, samples by block; the span is always yours
    block = rec.read(start=0, stop=10_000)                # [channels, samples], SI units
    volume, header = nn.io.read_nifti("t1.nii.gz")

The functions, their options and the readers are listed in [docs/reference.md](docs/reference.md), section "The Python face".

## JavaScript (WebAssembly, Node)

Every command of the command line is a function too, `nxr.<verb>.<noun>({ option: value })`, returning a Promise; arrays go in and come back as typed arrays with `shape` and
`dtype`, a refusal is an `NxrError` with the library's `.code`, and `nxr.open(dataset, mode)` gives a handle that keeps the dataset open. Built from the same C++ library as the
command (`bindings/wasm-io`; nothing is published to npm yet).

    import { nxrNeuro } from '@nxr/neuro'
    const nxr = await nxrNeuro()
    await nxr.create.dataset({ dataset: 'study/demo' })
    const psd = await nxr.read.statistic({ dataset: 'study/demo', subject: 'sub-01', name: 'rest_psd_welch' })   // psd.arrays.values: a Float32Array with .shape and .dtype

See [docs/reference.md](docs/reference.md), section "The JavaScript face".

## The command

One command, verb first: `nxr-neuro <verb> <what> [options]`.

    nxr-neuro --help                         # the verbs
    nxr-neuro run --help                     # the pipelines, by topic
    nxr-neuro run psd --help                 # one command: every option with its meaning, default and unit, and an example
    nxr-neuro run psd --dataset store/demo --subject S1 --recording rest --from 0:10 --to 1:10

| verb | what it does |
|---|---|
| `run <pipeline>` | every pipeline on a subject: the built-in computations (psd, crossspectrum, phasegradient, nulls, ...), the selection makers (`select_dyadic_surface`, `select_time_grid`, `select_bands`, `select_product`, `glue`), the one measuring definition `measure` with its presets, and the shipped definitions (`run global_bursts --band "8-16 Hz" --all`, `run gabor_bursts --band "8-35 Hz" --all`: bursts per channel matched by Gabor atoms of free width) and your own definition file (`run my.yaml ...`); a definition's parameters are options, each with its meaning, default and unit in `--help` |
| `compute <node>` | reserved for applying a node or a graph to arrays and files (not available yet) |
| `list datasets\|subjects\|recordings\|kernels\|results\|pipelines` | what exists |
| `list manifolds\|fields\|selections\|operators\|statistics\|dictionaries\|spaces\|acquisitions\|jobs` | the nodes of a subject, one description each, filtered and paged (`--name`, `--kind`, `--on`, `--limit`, `--after`, `--columns`) |
| `show dataset\|subject`, `read`, `query`, `export` | one thing in detail (`show dataset` counts each class of node; `read subject` describes every node of a subject, no arrays), stored arrays, measurements, a table for statistics |
| `import <what>` | from a Brainstorm protocol |
| `migrate measurements` | bring a dataset made before schema 53 to one measurement table (in place, resumable, `--dry-run`; the old statistics are kept) |
| `create`, `delete`, `compact`, `recover`, `add` | a dataset's and a subject's housekeeping (`delete subject --node CLASS:NAME` removes one node with what it owns and refuses one others depend on; `add field\|statistic --begin\|--append\|--commit` writes large values in pieces as one unit; `run` and `import` leave a job row, read by `list jobs`) (a group's reduce is `run roll_up`); `add` stores what was computed elsewhere (a field, selection, operator, line, product, dictionary, ...) |
| `edit manifold\|dictionary` | change a manifold's label, description and primary flag, or a dictionary's entries, without touching any array |

A dataset is a folder, named by its path with `--dataset <folder>`, and subjects `--subject <id>`. Results are written into the dataset with their provenance and a cache key, so a repeated
run with the same options is a cache hit. **A computation over a recording states its time range**: `--from <s> --to <s>` (seconds or `m:ss`) or `--all`; without one it refuses.
Recordings are streamed one chunk at a time with bounded memory whatever the range, and the range actually used is recorded with the result.

In a terminal the output is readable (a progress bar on stderr, a result line, what was stored); with `--json`, or when stdout is not a terminal, it is JSON lines, one per event,
unchanged. `--quiet` prints only the result line; colour only in a terminal and never with `NO_COLOR`. Man pages and shell completion for bash, zsh and fish are built with the command
(`nxr-neuro completion bash|zsh|fish`). The old spellings of the pipelines (`psd compute`, `compute psd`, `run DEFINITION --set NAME=VALUE`, ...) are refused with the new spelling named; the other old two-word spellings (`dataset list`, `convert subject`, ...) are refused with the new one named too (nxr-neuro is unreleased: no deprecation period). [docs/reference.md](docs/reference.md) lists every
command, its reference method, what it stores and what it cannot resolve; the grammar is in [docs/specs/2026-11-16-command-line.md](docs/specs/2026-11-16-command-line.md).

## Contributing

Read [ARCHITECTURE.md](ARCHITECTURE.md) and [CONTRIBUTING.md](CONTRIBUTING.md) first. Contributions are made under the project licence; there is no CLA.

## Licence

BSD-3-Clause, Copyright (c) 2026, the Neurodynamics-XR contributors: see [LICENSE](LICENSE). Third-party software and its licences:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Contributors are listed in [AUTHORS](AUTHORS).
