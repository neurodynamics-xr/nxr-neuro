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
| `io/` | `nxr_io` | the nxr dataset format (SQLite rows plus Zarr v3 arrays; a subject is a self-contained package, store format 53), a chunked recording reader, and a Brainstorm protocol importer |
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

## The command

    build/cli/nxr-neuro                # prints the usage: every verb with its options

Datasets are named `--dataset <datastore>/<name>` and subjects `--subject <id>`. Results are written into the dataset with their provenance and a cache
key, so a repeated run with the same options is a cache hit. Recordings are streamed one chunk at a time with bounded memory, and a computation always states its own time span.
(Per-verb `--help` is planned.) [docs/reference.md](docs/reference.md) lists every verb with its options, its reference method, what it stores and what it cannot resolve.

## Contributing

Read [ARCHITECTURE.md](ARCHITECTURE.md) and [CONTRIBUTING.md](CONTRIBUTING.md) first. Contributions are made under the project licence; there is no CLA.

## Licence

BSD-3-Clause, Copyright (c) 2026, the Neurodynamics-XR contributors: see [LICENSE](LICENSE). Third-party software and its licences:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Contributors are listed in [AUTHORS](AUTHORS).
