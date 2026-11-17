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
| `io/` (`io/formats/`: `nxr_io_formats`, pure file formats) | `nxr_io` | SQLite, zstd/blosc/zlib, matio, HDF5, nlohmann/json, rapidyaml (vendored; 2026-11-12). **Never `compute`** |
| `meeg/` | `nxr_meeg` | Eigen only. **No file I/O, never `compute` or `io`** (the pipelines hand it arrays) |
| `pipelines/graph/` | `nxr_graph` | `compute`, `meeg`, Eigen, nlohmann/json (amended 2026-11-17). **No file I/O, never `io`**: the graph executor, which knows arrays and parameters only |
| `pipelines/` | `nxr_pipelines` | `compute`, `io` (**only io's task API**, `nxr/io/task/`, plus the chunk service), `meeg`: **the only place they meet** |
| `cli/` | `nxr-neuro` | argument parsing and output only, with no maths and no SQL; it reaches data through io's task API |
| `bindings/` | MEX, Node, wasm, Python | every library (`compute`, `meeg`, `io`, `pipelines`), as `cli/` does: **faces only**. A binding converts arguments and results and calls the library, with no maths, no SQL and no logic of its own (amended 2026-11-17: the earlier "`compute` only" came from the old nxr-compute repository) |

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
6. **Enforced by** the dependency table, the layering guard tests (`cmake/check_layering.cmake`: recordings read only by the chunk service; no io
   include in `compute/` or `meeg/`; `layering_sql_only_in_io`: SQL, `database.h` and `nxr/io/records/` only inside io; `layering_nodes_without_io`: the registered nodes include no io header and hold no
   database, subject, dataset or file: what a node needs beside its inputs is bound by the runner and handed over as data, `nxr::pipelines::graph::Bound`), and the checklist in each PR.

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

## Decided structure (2026-11-17): data access, verbs, pipelines, faces

These rules complete the three layers above. Some of the code is still being migrated to them (see "Transition" at the end of this section); new code
follows them from the start, and no change may move existing code further from them.

**1. Data access: io in two levels** (spec `docs/specs/2026-11-17-io-record-api.md`).
- **Inside io: the repositories** (`nxr::io::records`: row structs, `Where` filters, read/list/find/create/update/remove), the shape of the Cortical
  Flow Studio's repositories. Only io, the binding that serves the Studio (`bindings/wasm-io`) and io's own tests and record tools use them.
- **For everyone else: io's task API** (`nxr/io/task/`), named by task, never as rows: **find** inputs (by kind, relation, parameters; one match or a
  clear error), **open** arrays for streaming, **commit** a result as one atomic unit (rows, arrays, provenance, cache key and measurements together, or
  nothing), **look up the cache**, **query measurements** (the read-time reduction over partitions, members and time).
- **Only io writes SQL or includes `database.h` or a `records/` header.** Pipelines and the command line never name a row, a table or a column.
- The SQLite engine is always the pinned amalgamation built with the project, never the system's.

**2. Two verbs, the same in every face** (spec `docs/specs/2026-11-17-compute-and-run.md`).
- **`compute`** applies a NODE or a GRAPH of nodes to data handed to it: arrays in Python and MATLAB; files in and out on the command line (with a
  provenance sidecar). It finds nothing and stores nothing in a dataset. What can be computed alone is exactly what can be chained in a definition: the
  list comes from the node registry.
- **`run`** runs a PIPELINE on a subject: it finds the inputs, streams the stated time range chunk by chunk, and stores the outputs with provenance and
  cache key. That is all it does: io around the graph.
- The command line, the Python wheel and the MATLAB binding have the same verbs and names (`nxr-neuro compute …` = `nxr.compute.…`; `nxr-neuro run …`
  = `nxr.run(…)`), plus `io` readers in the bindings.

**3. Every pipeline is a definition.**
- A pipeline is a YAML definition: its inputs, parameters (each with meaning, default and unit, from which `run <pipeline> --help` is generated),
  steps wired port to port, and outputs. Built-in pipelines ship as definitions in `pipelines/definitions/`; users write their own.
- **No pipeline is a hand-wired C++ chain.** A new analysis is new nodes (pure functions, registered with their ports, parameters and traits) plus a
  definition. Never a `<x>_compute` function that finds its own inputs, loops and stores.
- **A graph of pure functions is itself a pure function.** The graph executor (arrow and port checks, the halo composed from the nodes' traits, state
  carried across chunks as (block, state) → (output, state)) is pure and io-free (`pipelines/graph/`). The runner only finds, streams and stores.
- **A graph can be wrapped as a new node** (a composite, as TSL's `Fn` composes nodes), reusable in other graphs and by `compute`.
- A missing derived input the pipeline needs (e.g. a surface's eigenmodes) is made by its own run first, with a message saying what was made and why;
  never stored silently inside another run.

**4. Bindings are faces, exactly like the command line.** They may use every library layer, convert arguments and results, and call the library.
They hold no maths, no SQL and no logic of their own.

**5. Interoperability.**
- File formats are read and written natively in `io/formats/` (pure: bytes in, objects out; no store, no database). A library is used only when its
  licence is compatible with BSD-3-Clause and it is clearly easier than writing the format; GPL code is never copied or linked.
- Reference implementations (MNE-Python, nibabel, scipy, Brainstorm) only GENERATE synthetic goldens; they are never run-time dependencies.
- Readers parse every header field. What an import keeps is the user's choice: nxr-neuro is a toolbox and imposes no data policy on its users
  (`--anonymize` drops identifying fields). The repository itself never contains real or identifying data.
- Imported recordings are copied into the subject package by default; reading in place from the source file is an option.

**Transition** (to be removed when done): some pipelines are still hand-wired C++ verbs and some files still reach the database directly. They are
listed in the layering check's allowlist, which only shrinks. A hand-wired verb is converted into a definition; it is never extended. `run <name>`
dispatches to the hand-wired verbs until they are converted.

## Backends and node traits

Two cheap steps toward running the graph on a GPU. They change no result: the CPU is still the only backend, and every number is what it was.

**Every node declares its traits, where it is registered** (`NodeTraits`, `pipelines/include/nxr/pipelines/registry.h`):
- **Time behaviour**, one of:
  - `pointwise`: each timestep (frame, tile or event) is computed on its own, so any block split or block order gives the same values;
  - `windowed`: each timestep needs a halo or window around it (an FIR or FFT filter, the analytic signal, a Welch segment, an event's margin); the node states its window rule in `window_rule`;
  - `stateful`: carries state forward in time (a recursive filter, a state machine over frames, an accumulation that depends on the order of the blocks); its blocks must meet in time order.
- **Numeric precision**: the precision the node computes in, and what it outputs. The rule is that stored precision follows the input. Where a node computes in double on float32 input, `computes_double_on_float_input()` flags it. A node that outputs double whatever its input is (an accumulated statistic, a table of events) declares `float64` output.
- `uncertain` marks a classification that is a judgement the maintainers should confirm.
- The executor reads the declaration and refuses a node that has none. A new node must declare its traits (the test `test_node_traits` fails naming it). A `stateful` node is refused in a block-parallel context (`DispatchContext::block_parallel`); no executor sets that yet.

**One dispatch point.** The executor never calls a node's functions itself. It describes the call (`NodeCall`: the phase, the node's prepared state, its inputs) and hands it to `dispatch` (`pipelines/include/nxr/pipelines/dispatch.h`, `pipelines/src/dispatch.cpp`), which checks the traits and runs the phase on the context's backend (`Backend::cpu`). A GPU backend is added in that one place, without touching a pipeline, a node or the runner. Enforced by the test `dispatch_is_the_only_call_path` (`cmake/check_dispatch.cmake`), which fails when any other file calls a node's `prepare`, `start`, `chunk`, `chunk_at`, `fold`, `finish` or `run` through a step.

A new dependency, or a new modality folder (e.g. `pet/`), needs the maintainers' OK and a dated amendment in the table in `docs/reference.md`.
Namespaces follow the folder: `nxr::compute` and topic namespaces (`nxr::welch`, `nxr::filter`, `nxr::flow`, …), `nxr::io`, `nxr::bst`, `nxr::meeg`, `nxr::pipelines`.
