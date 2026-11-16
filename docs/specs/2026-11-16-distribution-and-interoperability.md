# nxr-neuro outside the project's machines: builds, HPC, interoperability 

Status: **findings and options** (opened 2026-11-16), not a plan yet. Decisions are recorded in the **Decision log** at the end as they are taken, each dated.

## Why

nxr-neuro is the project's engine for measuring the spatiotemporal patterns of brain activity (`README.md`). To serve the project's purpose beyond
a developer's workstation, it has to run where the data are (desktops and HPC clusters), be installable by people who are not its developers, and
hand its results to the tools the field already uses. Today it runs well from a source build, and its results are read by the
project's own tools only. This document records where it stands, how comparable software does it, and the options.

## 1. Where it stands (checked 2026-11-16)

### 1.1 One source, several faces

nxr-neuro is one C++ project with layered libraries (the dependency table in `CONTRIBUTING.md`): `compute/` (`nxr_compute`, the former
nxr-compute repo, brought in by `git subtree`; that repo is frozen), `io/` (the former nxr-io), `meeg/`, `pipelines/`, `cli/`, `bindings/`.
Like nxr-compute before it, one source builds several FACES, each per platform:

| face | host | reaches | state |
|---|---|---|---|
| `nxr-neuro` command | terminal, scripts, HPC jobs | everything (compute, io, meeg, pipelines) | built on macOS arm64; **never built on Linux** |
| WebAssembly `wasm-io` | the Studio (browser and desktop) | io + the analyses the Studio runs (its data client) | built; parity-tested against the Studio |
| WebAssembly `wasm` | browser | compute only | stale; nothing loads it (`docs/open-items.md`, Bindings) |
| MEX | MATLAB / Rheome | compute only | from nxr-compute's days |
| Node add-on | Node / Electron | compute only | from nxr-compute's days |
| Python module (wheel) | Python | would be everything, as the command | **does not exist** |

The dependency rule says bindings may depend on `compute` only. A Python (or fuller MATLAB) face that opens datasets and runs pipelines
needs it amended so a binding may use `pipelines`, as `cli/` does (a dated amendment the maintainers approves).

### 1.2 The binary

- One executable, about 19 MB (macOS arm64). Compiled in: Eigen, Spectra, geometry-central, HDF5, matio, c-blosc, nlohmann/json, rapidyaml.
- Linked at run time: the OS's `libsqlite3`, `libz`, `libc++`, `libSystem`, and **Homebrew's `libzstd`**: the one link that ties today's build
  to a machine with Homebrew. A distributed build links zstd (and, on Linux, sqlite and zlib) statically.
- Built only from source (`cmake -S . -B build -G Ninja && cmake --build build`, after `brew install zstd`).
- **No CI.** The GitHub Actions history in the log (MEX, wasm, Node add-on legs) came with nxr-compute's history; no workflow is in the tree.

### 1.3 Linux

Likely to build, not yet confirmed: C++17 and CMake; the heavy dependencies fetched or vendored; only SQLite, zlib and zstd from the system
(packaged by every distribution); `compute/src/parallel.cpp` already has `__linux__` paths. From source:

    sudo apt install build-essential cmake ninja-build libsqlite3-dev zlib1g-dev libzstd-dev
    git clone --recurse-submodules <nxr-neuro> && cd nxr-neuro && cmake -S . -B build -G Ninja && cmake --build build

Untested differences (gcc against Apple's clang) may surface. an HPC cluster is Linux, so the HPC case is the Linux case.

### 1.4 The command line

One binary, 43 commands, one shape: `nxr-neuro <noun> <verb> --dataset <datastore>/<dataset> --subject <S> [options]`. Output is JSON lines
(progress, then one `"stage": "result"` with `"cache": "hit" | "miss"`); a rerun with the same options is a cache hit, `--replace` recomputes;
recordings are streamed one chunk at a time (two chunks in memory) and `--chunks all` loops over the whole recording.

| group | commands |
|---|---|
| data and setup | `dataset create·list·show·delete·recover·compact`, `subject create·list·show·delete`, `convert list·scan·subject·surface` (Brainstorm only), `surface add`, `read` |
| source imaging | `noisecov`, `headmodel`, `inverse`, `pointspread` (`compute`), `lowpass` |
| geometry | `eigen solve`, `frames`, `tiles`, `patches` (`compute`), `pyramid spatial-bands` |
| time | `constantq`, `frametiling`, `joint` (`compute`) |
| spectra | `psd` (`--through` for a rotary pair), `sourcepsd`, `crossspectrum` (`--resolve bins`, `--estimator analytic`), `crossspectrum read`, `sourcetensor`, `hodgepower`, `speedspectrum` |
| patterns | `phasegradient`, `opticalflow`, `cyclepattern`, `quadrature`, `cellquadrature` |
| measurements | `pyramid build·reduce`, `measure compute·query·export`, `statistic compute` |
| nulls, reliability | `null`, `spatialnull`, `wishart`, `repeat`, `scaleselection` |
| pipelines | `run DEFINITION` (`run --list`: `global_bursts`, `band_bursts`, `cortical_cells`, …) |

What a new user lacks: per-command `--help` (the usage list and the comments in `cli/*.cpp` are the documentation); results named by long
strings copied from `subject show`; importing from anything but Brainstorm.

### 1.5 HPC

- The same binary; one job per subject or recording (a SLURM array), memory independent of the recording's length, `--threads` where a
  method uses cores. Format 52 keeps a subject in few files (small arrays in one SQLite file, large ones in ~64 MB shards), which cluster
  file systems need.
- **One writing job per datastore today:** the chunk service holds a queue lock (`.nxr-chunk-queue.lock`), so parallel jobs on one datastore
  wait for each other.
- **Designed, not built:** staging and merge (`docs/specs/2026-11-14-dataset-wide-runs.md` §2): each job writes a staging dataset on the node's
  local disk, then `dataset merge` folds them in, one transaction each. Not built yet.
- **On a cluster:** nxr-neuro is a command a workflow system calls. The pipeline that runs
  `nxr-neuro run` per subject, with the merge as its last step, does not exist yet.
- Group level after the per-subject jobs: `pyramid reduce`, then the group statistics.

### 1.6 Getting results out

| route | gives | limits |
|---|---|---|
| Python `nxr.io` | NumPy arrays + the database rows (`bundle`, `reader`, `array`, `q`) | axes, units and names are joined by the caller (`nxr.scopes` does it internally, not as a public API) |
| MATLAB `rheome.io.read` | MATLAB arrays (`zarr`) and readers for recordings, surfaces, kernels, Hodge packages, registrations | no statistics or measurements; reads array FOLDERS, so may not read format 52's packed small arrays (unverified) |
| nxr-neuro CLI | `read` (raw), `measure query` (JSON rows), `crossspectrum read` (JSON), **`measure export`** (measurement tables to a folder) | raw or JSON; one table exporter |
| the Studio | a reading or a measure saved as CSV | what is on screen |
| `nxr.convert` (Python) | **imports only**: Brainstorm, FreeSurfer, MNE EEG/MEG, BIDS-iEEG | nothing exports to MNE, BIDS derivatives, GIFTI, NIfTI, NetCDF |

### 1.7 Three toolboxes doing similar things

nxr (Python, v1.17, `pip`-installable, its own `nxr` command), Rheome (MATLAB) and nxr-neuro (C++) duplicate many analyses; nxr keeps
"Python twins" of nxr-neuro verbs (e.g. `nxr.pipelines.opticalflow`, which holds its window in memory where nxr-neuro streams it).

## 2. How comparable software does it

| | FreeSurfer | FSL / AFNI | MRtrix3 | Connectome Workbench | MNE-Python | fMRIPrep (a BIDS App) | **nxr-neuro** |
|---|---|---|---|---|---|---|---|
| interface | many programs (`recon-all`, `mri_*`, `mris_*`) | many programs; AFNI writes analysis scripts | many programs, one consistent style | **one binary, subcommands** (`wb_command -<op>`) | a Python library | one container, one command | **one binary, subcommands** |
| data | own folder tree, standard files (MGZ, surfaces, labels) | NIfTI | `.mif` + NIfTI | **GIFTI, CIFTI, NIfTI** | FIF + objects | BIDS in → BIDS derivatives out | **own database + zarr datastore** |
| provenance, caching | logs, timestamps | headers, scripts | headers | — | object info | reports, sidecars | **built in; reruns are cache hits** |
| long recordings | n/a | n/a | n/a | n/a | in memory or memory-mapped | n/a | **streamed by chunks** |
| help | `-help` | `-help` | excellent `-help` | `-help` | API docs, tutorials | docs, reports | **one usage list** |
| install | large tarball/package + licence file + environment | installer / conda | conda, Homebrew, apt | zip of binaries per platform | pip / conda | Docker / Apptainer | **source only** |
| HPC | job per subject | job per subject (`fsl_sub`) | job per subject | job per operation | scripts | **the model**: `fmriprep bids/ out/ participant --participant-label 01` in Apptainer | job per subject; staging/merge designed |

(From general knowledge of these tools, not re-checked against their current documentation.)

**Where nxr-neuro stands.** What it measures (travelling, rotating and standing patterns in physical units, the instrument's resolution
built in) no other tool here does. Its datastore gives provenance, caching, streaming and the Studio. Its interface style is Workbench's.
**The main gap is interoperability:** every other tool trades in standard files other tools open; nxr-neuro's results are read only by the
project's tools.

## 3. Lessons worth taking

1. **Per-command help**: `nxr-neuro <command> --help` with options, defaults, units and an example (MRtrix, Workbench). Cheap and expected.
2. **Export to standard formats**, as a way OUT of the datastore, never a replacement for it: cortical maps → **GIFTI/CIFTI** (open in
   Workbench and freeview); tables → **TSV**; sensor-level results → **MNE objects / FIF** (`Spectrum`, `CrossSpectralDensity`); labelled
   arrays → **NetCDF/Zarr via xarray**; a study → a **BIDS-derivatives** layout. Reuse the libraries that write these (nibabel, MNE, xarray,
   pandas) rather than writing writers.
3. **Import from standard formats** in nxr-neuro itself: BIDS (MEG/EEG), FIF, FreeSurfer subjects (some exist in nxr, Python only).
4. **A BIDS-App entry point** for clusters, fMRIPrep's pattern: `nxr-neuro bids/ out/ participant --participant-label 01`, in an Apptainer image.
5. **Ordinary install channels**: release binaries per platform, `pip`/`conda`, a container. Signed and notarised on macOS.
6. **Confirm Linux** and keep it confirmed: CI that builds and tests on Linux and macOS and produces the release binaries and wheels.
7. **Staging and merge** for parallel jobs on one dataset, then a **job pipeline** that runs `nxr-neuro run` per subject on a cluster.
8. **A labelled read API**: stored results as xarray `DataArray`s (named dimensions, coordinates, units and provenance in `attrs`) and
   measurement queries as pandas `DataFrame`s; most exports then come from those libraries.

## 4. Options for the faces

### 4.1 A Python face (a wheel)

nxr-neuro bound to Python with **nanobind** (zero-copy Eigen ↔ NumPy, the stable ABI so one wheel per platform covers Python ≥ 3.12), built by
**scikit-build-core**, wheels made by **cibuildwheel** in CI for macOS arm64 and Linux x86-64 (manylinux). Three levels:
- **pipelines**: each verb with its options struct as keyword arguments (the method rule already makes the names match the CLI), returning
  what it stored;
- **reading**: the labelled read API (lesson 8);
- **compute**: the pure functions on NumPy arrays (the architecture already requires that they take a matrix made anywhere).

Errors become Python exceptions by code; progress is a Python callback; the C++ releases the GIL while it runs; memory is unchanged (results
stored, read back as needed). `pip install nxr-neuro` would install both the module and the `nxr-neuro` command. Hosted as GitHub release
assets or a private index (the repos are private); a local wheelhouse on the cluster.

### 4.2 nxr on top of nxr-neuro

nxr-neuro the one engine; nxr the Python face (binding, reading, plots, group statistics, prototyping); Rheome the MATLAB face via MEX, or
frozen as a reference; the Studio unchanged (WebAssembly).
- **Easier:** one implementation a method (no parity drift); chunked streaming and C++ speed for Python users; one reader of the format.
- **Harder:** compiled wheels a platform; prototyping in C++ is slower (rule: a method may start in nxr, NumPy, and moves to nxr-neuro once
  stable, its NumPy version then the golden); debugging across the boundary; a migration (inventory the duplicates, retire twins one by one).

### 4.3 A desktop user

Install: a release binary, Homebrew tap (`brew install neurodynamics-xr/tap/nxr-neuro`), `pip`, later conda-forge. Use: `dataset create` with
the templates, `convert`, the compute verbs, the Studio desktop app (File → Change datastore) to look, Python or `measure export` to analyse.
Needs: the templates (fsaverage, modes) as a download within their licences; importers beyond Brainstorm; a quickstart on a synthetic or
public dataset (never a restricted cohort).

## 4.4 The graphical face: the Cortical Flow Studio (added 2026-11-16)

Other suites ship a viewer beside their tools: FreeSurfer's freeview, FSL's FSLeyes, Workbench's `wb_view` beside `wb_command`, AFNI's GUI.
The project's is more than a viewer: **the Studio computes** (interactive chains on the GPU, the measures, the lag map). The closest analogues are
**Brainstorm** (a GUI that runs whole analyses, with scripting) and the **`wb_view` + `wb_command`** pairing (a GUI and a CLI on the same data).

**Where it stands** (the Studio's `docs/INSTALL.md`, `app/electron-builder.yml`, checked 2026-11-16):
- **Web** (corticalflow.app): no install; a served datastore is read-only; no native compute.
- **Desktop** (Electron): macOS `.dmg` (notarisation script in the build), Windows installer (NSIS; verified on Windows 11, 2026-08-17), Linux
  `AppImage` configured. It owns a writable datastore: prepare, convert, compute.
- **Its compute comes from three places:**
  - its own TypeScript/GPU chains (the interactive views);
  - `wasm-io` from nxr-neuro (the data client and some analyses);
  - on desktop, the **Node add-on of the frozen `nxr-compute` repo** ("[native] nxr-compute addon loaded"). That is an engine nxr-neuro has
    replaced: its `compute/` is that code, moved on.
- nxr-neuro's pipelines are written as CPU replicas of the Studio's GPU chains and parity-tested against them (`CONTRIBUTING.md`), so the Studio
  and nxr-neuro already share their definitions of the methods.

**What a GUI face of the one engine would mean:**
1. **One install, GUI and CLI:** the desktop Studio bundles the `nxr-neuro` command (as FreeSurfer ships freeview with its tools), so a user who
   installs the app can also script and batch.
2. **The Studio's native compute is nxr-neuro:** its Node add-on built from nxr-neuro (`bindings/node`) instead of the frozen nxr-compute.
   Heavy and stored computations (a whole recording, a pyramid, a dataset) run through nxr-neuro: natively on desktop, WebAssembly in the
   browser. The GPU chains stay for what must be live (scrubbing, the seed lag map), under the parity tests that already exist.
3. **Every GUI computation reproducible as a command:** a computation started in the Studio is a `nxr-neuro run` definition (or verb) with its
   options, shown and copyable, so a result made by clicking can be rerun in batch or on a cluster. This is Brainstorm's pipeline editor and AFNI's
   generated scripts. Both the Studio's "recipes" (stored pipeline parameters) and nxr-neuro's definitions exist; this joins them.
4. **The GUI also opens standard files** (GIFTI, CIFTI, NIfTI) and exports what it shows (lesson 2), so the Studio is useful beside other
   suites, not only on our datastore.

## 4.5 Two Studios from one code: the web one to show, the native one to compute (added 2026-11-16)

Framing (2026-11-16): the Studio's portability (browsers, embeddings) is a strength for sharing, demonstrations and illustrations, and the web
Studio need not be a computation engine like a native Studio.

**What exists** (the Studio, `main`, checked 2026-11-16): the web app (corticalflow.app) is **read-only by construction** ("a served datastore
cannot create an eigenbasis, a mass matrix or a Dirac fusion, so everything it shows was prepared beforehand"); `/browse` lists what is
published; `studio/src/studio/embed/` embeds a scene (`show/scene.ts`) with only the controls it lists. Desktop releases come from a separate
public repo (`cortical-flow-releases`).

**The split, made deliberate:**

| | web / embedded Studio | native (desktop) Studio |
|---|---|---|
| purpose | share, demonstrate, illustrate, teach, a figure in a paper or talk | analyse: prepare, convert, compute, store |
| data | a **published package**: read-only, static files, only what its views need | a writable datastore |
| compute | **only what interaction needs, on the GPU**: scrubbing, colouring, filtering a stored series, the seed lag map from a STORED cross-spectrum, reading stored measures at any scale | everything, through nxr-neuro (D10, D11) |
| install | none: a link or an embed | the app (and with it the `nxr-neuro` command) |
| engine weight | small and fast to load: no heavy WebAssembly compute | full |

**What it entails:**
1. **Publish, from the native Studio or the command line.** A step that makes a self-contained, read-only package of chosen results and
   views: the surfaces, maps, statistics, measurements and the slice of a recording a view needs (not the whole dataset), plus the scenes
   (`show/scene.ts`) that open them. Static files, so it can be hosted anywhere (a project server, GitHub Pages, a journal's supplementary material).
   Format 52 helps: a subject's small arrays are one SQLite file, read over HTTP with ranged requests.
2. **Provenance travels with it.** Each published result carries its producer and options and the `nxr-neuro` command that made it (D12), so a
   reader of a paper can see, and a collaborator can rerun, how a figure's numbers were made.
3. **A data-protection gate.** Publishing makes data public, so it refuses restricted or identifiable data (for example a
   restricted cohort never goes on a public page), unless a dataset is marked publishable. The
   dataset's row says which.
4. **A capability profile, not two codebases.** One Studio code, built in two profiles: the web build leaves out what needs a writable store
   or heavy compute and loads only the GPU chains interaction needs. The capability is detected (a served store, no native engine) and the UI
   offers only what that profile can do, as the web app already refuses eigenbasis creation.
5. **Interactive compute in the browser stays GPU, under the existing parity tests** with nxr-neuro, so what a reader sees live on a published
   page is the same method as the stored result.
6. **Embedding as a product:** a scene embedded in a web page, a slide, a docs article (the `embed` layer exists), with a size budget so it
   loads quickly, and a still image fallback for print (the snapshot path `nxr.snapshot` already renders).

## 4.6 Looking at data and results: from the command line, and in notebooks (added 2026-11-16)

**After computing with the command line** (on a Linux desktop, or on a cluster without a display):

| way | how | state |
|---|---|---|
| the native Studio | open the datastore (File → Change datastore) | the desktop app exists (Linux `AppImage` configured, not checked on Linux) |
| **a local viewer served from the command** | `nxr-neuro view --datastore D [--port P]` serves the datastore read-only and opens the web Studio on `localhost`, the TensorBoard/Jupyter pattern; on a cluster through an SSH tunnel (`ssh -L`), so results are looked at where they are, nothing copied | **does not exist**; the web Studio and its read-only served mode do (4.5), and `nxr.view` already serves a bundle from a local HTTP server for the viewer |
| figures without a display | `nxr.snapshot` (PNG/PDF, headless) and `nxr.scopes` (matplotlib) | exist, in nxr (Python) |
| numbers | `nxr-neuro read`, `measure query`, `measure export` | exist |

**In a notebook, with the Python face (4.1):**
- **A chunk, as a labelled array**, through the chunk service, as the command reads it (never the whole recording by default):
  `X = sub.recording(R).chunk(0)`: an xarray `DataArray` (channel × time; channel names, seconds, units, bad spans).
- **Timeseries:** `nxr.show.timeseries(t, X, mode="stacked")` (exists, matplotlib: stacked, overlay, parade, raster), or, to scroll, the same
  chunk as an `mne.io.RawArray` and MNE's browser (`raw.plot()`), reusing MNE rather than writing a browser.
- **Animated source mapping:** the source estimate of the chunk (the kernel applied to it, a compute call; looking needs nothing stored) shown
  on the cortex over time. Three ways, one renderer:
  - **interactive in the notebook**: the web Studio's viewer as a notebook widget (anywidget), handed the surface and the frames, with time
    slider and play: the presentation tier (4.5) inside Jupyter. Does not exist; `nxr.view` today opens the same viewer in a browser tab
    from Python, with `Time`, `Play`, `Speed` (the twin of `rheome.view`);
  - **a movie** for talks: `nxr.snapshot` frames → MP4/GIF;
  - **a still** for print: `nxr.snapshot`.
  A second renderer (PyVista, MNE's `stc.plot`) is possible but would draw the cortex differently from the Studio and the figures.

## 5. Open decisions (for the maintainers)

| # | decision | options | recommendation |
|---|---|---|---|
| D1 | Converge: nxr-neuro the one engine, nxr and Rheome its faces? | yes, gradually · no, keep three implementations | yes, gradually (4.2), starting with an inventory of the duplicates |
| D2 | Rheome's role | MATLAB face via MEX (kept current) · frozen as a reference | depends on who works in MATLAB |
| D3 | A Python face (amends the dependency rule; adds nanobind, scikit-build-core) | build it (4.1) · not now | build a first slice: `open`/`read`, `psd`, `crossspectrum`, a few compute functions; one wheel each for macOS and Linux |
| D4 | Linux and CI | confirm now · later | now: a CI job that builds and tests on Linux and macOS (lesson 6) |
| D5 | Distribution channels | release binaries · Homebrew · pip · conda-forge · container | release binaries + pip first; container with D7 |
| D6 | Interoperability: exports and imports | which formats first | exports first: GIFTI (cortical maps), TSV, MNE objects; then BIDS-derivatives; imports from BIDS MEG/EEG and FIF |
| D7 | HPC | staging/merge + a job pipeline · BIDS-App container · both | staging/merge + a job pipeline first; the BIDS-App container for outside users |
| D8 | Per-command help | yes · no | yes (lesson 1) |
| D9 | Who outside the project it is for, and when | project only for now · collaborators · public release | needed to order D3–D8 |
| D10 | The Studio's native engine | keep the frozen nxr-compute add-on · nxr-neuro's add-on | nxr-neuro's (4.4 point 2): one engine, and the frozen repo stops being a dependency |
| D11 | One install: the desktop Studio bundles the `nxr-neuro` command | yes · separate installs | yes (4.4 point 1) |
| D12 | GUI computations reproducible as commands | yes · not now | yes (4.4 point 3): the Studio shows the `nxr-neuro` command / definition of what it computed |
| D13 | The web Studio's role | a presentation tier (read-only, published packages, GPU interaction only) · a compute engine too | presentation tier (4.5), as the maintainers framed it |
| D14 | Publishing | a publish step (package + provenance + data-protection gate) · ad hoc | a publish step (4.5 points 1–3), from the native Studio and the command line |
| D15 | Viewing from the command line | `nxr-neuro view` serving the web Studio on localhost (SSH tunnel on a cluster) · native Studio only | `nxr-neuro view` (4.6): no install beyond the command; works on a remote cluster |
| D16 | Viewing in notebooks | the Studio's viewer as a notebook widget · a second renderer (PyVista/MNE) | the Studio's viewer as a widget (4.6): one renderer for app, web, notebook and figures |

## 6. Publication and licence (decided 2026-11-16)

nxr-neuro is published as a new public repository whose first commit is a cleaned tree, so GitHub Actions (free and unlimited on public repositories) can
carry the CI. The previous private repository stays as the archive of the history. The tree contains no restricted-cohort data or identifiers, no private
paths, no results measured on restricted datasets, and no internal process notes; internal instructions for the maintainers' own tools live outside it.

- **Licence:** BSD-3-Clause, `Copyright (c) 2026, the Neurodynamics-XR contributors` (`LICENSE`, `AUTHORS`). No CLA and no DCO: contributions are made
  under the project licence, as in MNE-Python and the scientific Python stack. nxr-compute carries the same licence.
- **Third-party software:** `THIRD_PARTY_NOTICES.md` lists every vendored or fetched dependency with its licence (Eigen and Spectra MPL-2.0,
  geometry-central MIT, pocketfft BSD-3-Clause, nlohmann/json and rapidyaml MIT, HDF5, matio, c-blosc and zstd BSD-style, zlib, SQLite public domain).
- **Contributor documents:** the engineering rules are public (`ARCHITECTURE.md`, `CONTRIBUTING.md`); `CLAUDE.md` only points to them.
- **Real-data checks** read their data from environment variables (`NXR_FIXTURE_DATASET`, `NXR_FIXTURE_PROTOCOL`) and are skipped when unset.

## 7. The command line (opened 2026-11-16)

**How it works.** `nxr-neuro` is the executable built from `cli/` (C++, ~2,900 lines, its own small parser `cli.h` `Args`). The shell finds it on
the PATH and passes the words; `nxr_neuro.cpp` dispatches by the first words to a `*_command.cpp`, which parses the options into the verb's
options struct and calls the library (`pipelines/`: e.g. `psd_compute(dataset, subject, options, progress)`), then prints JSON lines. By the
architecture rule, `cli/` only parses and prints, so the command is a front end: its name and grammar can change without touching the
library, the stored data or the Studio; a Python binding would call the same library functions.

**What is unergonomic today:** `--dataset D/N --subject S` on every command; long stored names copied from `subject show`; JSON always; no
per-command help or shell completion; a mixed grammar (`psd compute`, `measure query`, `lowpass`, `read`, `run X`); nxr (Python) also installs
an `nxr` command.

**Options** (from git, kubectl, MRtrix, Jupyter):
1. **Help** (DECIDED, below): `nxr-neuro <cmd> --help` with each option's meaning, default and unit (from the options structs' comments, so help
   cannot drift from the code), an example; generated man pages and shell completion (bash, zsh, fish).
2. **Context instead of repeated flags:** `nxr-neuro use <dataset> --subject S`, or found from the current folder, or `NXR_DATASET` /
   `NXR_SUBJECT`. Results still record the full dataset, subject and options in their provenance.
3. **Short names:** aliases (`rest1`, `mn`) or unique prefixes for recordings, kernels and results, with tab completion of the real names.
4. **One grammar, verb first:** `nxr-neuro compute psd rest1`, `run global_bursts rest1`, `list recordings|kernels|results`, `show <result>`,
   `export <result> --to tsv|gifti|...` (section 3), `view` (4.6).
5. **Readable by default, JSON on request:** tables and progress bars in a terminal; with `--json` or when piped, today's JSON lines unchanged
   (existing scripts keep working).
6. **The name:** `nxr-neuro` now; `nxr` later, together with convergence (D1), when the Python package stops shipping its own `nxr` command;
   both names during a transition. Check the names on PyPI, conda-forge and Homebrew before going public.
7. **How:** stay in C++ (one binary, instant start, nothing to install on clusters) with a header-only argument library such as CLI11
   (subcommands, `--help`, validation, completion): a new dependency (the maintainers' OK). Only `cli/` changes.
8. **A rule kept:** method parameters never come from hidden defaults or config files; only paths and the context may.

## Decision log

| date | decision | follow-up |
|---|---|---|
| 2026-11-16 | Framing for D13: the web Studio is for sharing, demonstration and illustration; it need not be a computation engine like the native Studio | section 4.5; D13 itself still to confirm |
| 2026-11-16 | nxr-neuro becomes public: a new clean public repository, the current one kept as the archive | section 6 |
| 2026-11-16 | Per-command `--help` (section 7, point 1) | in the CLI work |
| 2026-11-16 | The command name stays `nxr-neuro`, with the verb-first grammar (section 7, points 4 and 6); possibly `nxr` at convergence (D1) | in the CLI work, with `--help` |
| 2026-11-16 | Nothing lab-internal in the public repository | engineering rules in public contributor documents |
| 2026-11-16 | Copyright holder: the Neurodynamics-XR contributors | `LICENSE`, `AUTHORS` |
| 2026-11-16 | Licence: BSD-3-Clause, no CLA and no DCO; the same for nxr-compute | `LICENSE` in both repositories |
