# Architecture of nxr-neuro

nxr-neuro is a C++ toolkit for measuring the spatiotemporal patterns of cortical activity from MEG/EEG: the nxr dataset format, the maths on the cortex,
MEG/EEG source imaging, the pipelines that join them, and one command, `nxr-neuro`. This document is the architecture every change must follow. It is
the first thing to read before contributing; the practical workflow (building, testing, adding a verb, style) is in `CONTRIBUTING.md`, and every verb and
method is described in `docs/reference.md`. "The Studio" below is the Cortical Flow Studio, the project's TypeScript/WebGPU application, whose engine the
pipelines mirror; "nxr" is the project's Python reference implementation.

## The architecture is paramount

**Every change follows the architecture below. It is never broken. If a change seems to need breaking it, stop and ask the maintainers before writing any code;
a workaround, a "temporary" exception or a shortcut is a breach.** Every other rule in this file and in CONTRIBUTING.md applies inside it.

Four parts, as the Studio's GPU engine has them (resident store → device plan → passes / compute nodes → readback):
1. **Data loading (io).** Recordings come as blocks through the chunk service; stored nodes (kernels, surfaces, cross-spectra) are read by name.
   Nothing is computed here.
2. **Preparation and orchestration (host, `pipelines/`).** Select and match the inputs (e.g. the MEG channels a kernel uses, in its order); TYPE-CHECK
   everything before any data flows (field types ℝ/ℂ/ℍ, shapes, units, channels against kernel, surface against kernel, each node's output against the
   next node's input: a mismatch is refused with its reason); build the graph from the caller's request; bind the constants once (kernel, mesh
   operators, normals).
3. **The pipeline: a GRAPH of pure functions** (the Studio's GPU/TSL chain). Each node is a pure function of matrices with typed input and output
   ports (in `compute/`, `meeg/`; the same function a test calls directly). Edges carry data from one node's output to the next node's input; per block
   the data flows through the graph. Nodes are POINTWISE: a reduction across time or space is never a node's own; it is a measurement over a
   partition (the cardinal rule below). The graph knows nothing of files, chunks or subjects, and decides nothing about which data it is given.
4. **Readback → io.** Outputs are gathered and either materialized (stored as Fields: frames, series, source maps) or reduced by the measurement
   functions over partition Selections (time and/or space), and stored by io with their provenance (the graph and its inputs), or handed to the next
   analysis.

**REDUCTIONS BELONG TO THE SELECTION SYSTEM.** This keeps the whole system modular and
composable.
- **A compute chain produces POINTWISE outputs:** values per element (vertex, channel) per sample (frame, segment). No step sums, averages or otherwise
  reduces across time or space on its own.
- **At the end of a chain, its outputs are either MATERIALIZED or REDUCED:**
  - materialized: stored as Fields (time frames, a time-resolved source map, a series);
  - reduced: through **partition Selections and the measurement functions**, the one reduction system (the Studio's `measureOverPartition`, the
    measure registry, the lookup).
- **Time is partitioned as space is.** A chunk is a time partition (the chunk grid `<recording>_chunks` is a Selection), as are a window or a frame
  tiling. A reduction over time, over space, or both is a measurement over a partition, never a bespoke loop or accumulator.
- **A ratio of reduced quantities** (a power-weighted phase gradient = mean phase current / mean power; a speed; a travelling index; a normal share) is a
  **derived measure**, computed at read from its stored ingredients, as the lookup derives `sd` from `mean` and `mean_square`. Never a reduction of the
  ratio itself.
- **Names say what a function computes** (`phaseCurrent`, `power`, `phaseGradient`), never how the code is organised (`sums`, `readout`).

The detailed rules are below: the dependency table, **the architecture policy** (what each layer may decide), **pipelines as replicas of the Studio's GPU
chains**, the chunk service, and the PR checklist. Design a new pipeline as its graph first (nodes, ports, inputs, outputs), check it against the
Studio's chain, and only then write code.

## Layout and the dependency rule (never broken)

| folder | library | may depend on |
|---|---|---|
| `compute/` | `nxr_compute` | Eigen, Spectra, geometry-central, pocketfft (vendored). **No file I/O, never `io`.** Its own guide: `compute/CONTRIBUTING.md` |
| `io/` | `nxr_io` | SQLite, zstd/blosc/zlib, matio, HDF5, nlohmann/json, rapidyaml (vendored; 2026-11-12). **Never `compute`** |
| `meeg/` | `nxr_meeg` | Eigen only. **No file I/O, never `compute` or `io`** (the pipelines hand it arrays) |
| `pipelines/` | `nxr_pipelines` | `compute`, `io`, `meeg`: **the only place they meet** |
| `cli/` | `nxr-neuro` | argument parsing and JSON output only, with no maths |
| `bindings/` | MEX, Node, wasm | `compute` only |

## The architecture policy: three layers, each knowing only its own job

Code that needs to know another layer's business is in the wrong place. **Compute** transforms matrices, **data** loads and stores, **orchestration**
composes what the caller asks for. The dependency table says what a library may import; this says what each layer may decide.

1. **Compute (`compute/`, `meeg/`): functions of matrices.**
   - In: arrays and the parameters of the maths. Out: arrays. Nothing else.
   - A function never reads or writes a file or a row, never knows where an input came from (a recording, a stored cross-spectrum, a kernel, a
     synthetic array), never knows about chunks, spans or the run's time window, and never picks its own input.
   - No mode options that only choose inputs or time handling (`signal=normal|ambient`, `mode=frames|average`): the caller passes a different input,
     or calls the function more times.
   - No function reduces across time or space: reductions are measurements over partition Selections (the cardinal rule, top of this file).
   - General algebra stays general (e.g. A M Aᴴ on a sparsity pattern), its own function, never buried inside a domain function.
   - The test: the function can be called with a matrix made anywhere.
2. **Data (`io/`): loading and storing.**
   - Recordings are read only through the chunk service (span, halo, chunks, the job). Stored nodes are read by name; results are written with their
     provenance and cache key.
   - Data never does maths beyond decoding, and never decides what a run computes.
3. **Orchestration (`pipelines/`, `cli/`): composition only.**
   - A pipeline is a chain: named inputs (from io) → compute steps → named outputs (to io).
   - Which inputs and which steps are the caller's request, named on the command line. A pipeline never hard-wires a source ("always the stored sensor
     cross-spectrum"), a signal, or a span.
   - Orchestration never loops over samples or chunks (the chunk service's job), never does maths (compute's), never knows file layouts (io's).
   - Defaults exist only for a method's reference parameters (the method rule: Brainstorm's bands, Welch's window), never for which data is used.
   - The command line parses and prints. It composes nothing the library doesn't.
4. **Port, don't redesign.** Before building any loading, looping, orchestration or compute structure, find the Studio's (and nxr's) design of it and
   port it, with its names.
5. **Before every PR, each answer must be yes** (and the PR description says so):
   - Can each new compute function be called with a matrix from anywhere?
   - Can the pipeline run the same chain on another input of the same kind just by naming it?
   - Is every loop over samples or chunks, and every file access, inside io?
   - Is every choice of which data (signal, source, span) made by the caller, not inside a function?
6. **Enforced by** the dependency table, the layering guard test (`cmake/check_layering.cmake`: recordings read only by the chunk service; no io
   include in `compute/` or `meeg/`), and the checklist in each PR.

**Pipelines are CPU replicas of the Studio's GPU/TSL chains.** Design every pipeline as if it were one of the Studio's
device chains, so nxr-neuro can later supply data to them, or run them, with no redesign.
- **Host (the Studio's CPU side): loading and orchestration.** The chunk service (residency, the chunk loop, the halo) delivers blocks; the
  orchestration picks the chain from the caller's request and runs it block by block.
- **Device (the Studio's GPU side): the chain.** An ordered series of pure stages, each a pass over a block (a chunk plus its halo), as the Studio's
  `SignalChain` (`studio/src/engine/toolbox/signal/chain.ts`) and its device form (`studio/src/studio/engine/gpu/deviceChain.ts`):
  - members (a linear spatial map) → filter (zero-phase sections, a band, a Meyer or Morlet wavelet band) → transform (none · envelope ·
    wavelet power or complex · spectrum, with the Studio's output shapes `xt`, `xft`, `xf`);
  - a linear map (a kernel, a form's operator) reads the chain BEFORE any non-linear transform (D106);
  - then the field's forms (`fieldForms.ts`) and the measures (reductions as sums folded across blocks).
  - Stages keep the Studio's order and names; state carried between blocks is an accumulator, as a device chain's is.
- **Readback (gather):** the chain's outputs come back to the host, which stores them (io) or hands them to the next analysis.
- A new pipeline is first written as its chain (stages, inputs, outputs) and checked against the Studio's chain before any code.
- The chain is a GRAPH of pure nodes with typed ports (the paramount architecture, top of this file); its executor is generic: it validates types,
  binds constants, runs the nodes over each block, folds reducing nodes and gathers outputs, with no analysis-specific code.

**Recordings are read only by io's chunk service.**
- `io/include/nxr/io/chunk_loop.h` is the one place a recording is loaded and looped over: the span of a run (always stated by the caller; never defaulted), whole chunks read in order, one at a time with bounded memory,
  the rows asked for, the bad-sample mask, progress.
- A pipeline only declares what it reads (the recording, its rows, its span: `span_of`, `source_of`) and passes a `visit` that folds each chunk into its accumulators.
  It never opens a recording's array, loops over its samples, or works out chunks.
- Compute (`compute/`, `meeg/`) receives a data matrix and never knows its size or where it came from.
- Enforced by the test `layering_recordings_read_by_chunk_service` (`cmake/check_layering.cmake`). A new recording reader goes through the service, never around the guard.
- Before building any io, loading or orchestration piece, look for the Studio's (and nxr's) design of it first and port it. Do not design it again here.

A new dependency, or a new modality folder (e.g. `pet/`), needs the maintainers' OK and a dated amendment in the table in `docs/reference.md`.
Namespaces follow the folder: `nxr::compute` and topic namespaces (`nxr::welch`, `nxr::filter`, `nxr::flow`, …), `nxr::io`, `nxr::bst`, `nxr::meeg`, `nxr::pipelines`.
