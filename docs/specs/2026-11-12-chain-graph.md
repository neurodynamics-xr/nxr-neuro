# Pipelines as graphs of pure steps (spec, 2026-11-12; APPROVED by the maintainers the same day; amended the same day: reductions are the Selection system's)

**Why.** The architecture is paramount (`CONTRIBUTING.md`, top): loading (io) → preparation and type checks → a graph of pure functions → readback to io.
nxr-neuro's pipelines are CPU replicas of the Studio's GPU chains, so that nxr-neuro can later supply data to those chains, or run them, with no
redesign. Today each verb (`sourcepsd`, `csd`, `sourcetensor`, `phasegradient`, …) hand-wires its own chain inside one C++ function. This spec
replaces that with **definitions run by one generic runner**, as nxr already does.

**Nothing here is invented.** Two existing designs are ported:
- **The graph and its runner come from nxr** (`nxr/docs/pipelines.md`): a pipeline is a **definition**. It has inputs found through the data model,
  parameters, **steps wired port to port**, and outputs. A runner plans it (arrows checked before any data is read), runs it chunk by chunk, and
  stores its outputs.
- **What each step computes comes from the Studio's GPU passes and their CPU twins** (`studio/src/studio/engine/gpu/`, `engine/toolbox/signal/chain.ts`,
  `engine/toolbox/differential/fieldForms.ts`):
  - the `SignalChain` order (members → filter → transform), and D106 (a linear map reads the chain before any non-linear transform);
  - the field types ℝ/ℂ/ℍ and their forms;
  - the halo (`HALO`, reflected only at the recording's ends);
  - the resident store's split: an operator uploaded once, the block per chunk.

  The Studio has **no generic node or executor** on the GPU side: its passes are hand-wired, each with a CPU twin as its parity reference. Its archived
  node-graph design (`docs/archive/superpowers/specs/2026-07-22-node-graph-editor-design.md`: `NodeDefinition`, `PortDef`, `canConnect`, a registry,
  a topological runner) supplies only vocabulary, and agrees with nxr's.

## Amendment: reductions belong to the Selection system

`ARCHITECTURE.md`'s cardinal rule applies to this spec, and it supersedes every part below that has a step reduce on its own.
- **Steps are pointwise.** A step computes per element per sample and never sums or averages across time or space.
- **nxr's `accumulate` kind is served by the measurement functions over a TIME partition.** A chunk is a member of the chunk grid
  (`<recording>_chunks`), and a window or a frame tiling is a partition too. The runner folds a chain's outputs into measurements over that partition
  (the Studio's `measureOverPartition`: weighted sums that merge exactly), not into bespoke accumulators.
- **At the end, outputs are materialized** (time-resolved Fields: frames, series, source maps) **or reduced** over partitions (time, space or both).
  Ratios are **derived measures** computed at read.
- **The steps change accordingly:**
  - `mean_square` becomes the pointwise `power` (|x|²), reduced as the measurement `mean` over the time partition;
  - `second_moments` becomes the pointwise outer product J Jᴴ (hermitian3), reduced as a mean over the time partition (the band tensor);
  - the phase "sums" and "readout" become `phaseCurrent` (j = Im(z̄∇z), the phase current), `power`, `gradientPower` (|∇z|²), and the pointwise maps
    `phaseGradient` (j / |z|²), `phaseVelocity` and `travellingIndex`. Over a span or a region, the phase gradient, speed and travelling index are
    derived measures of the reduced j, |z|² and |∇z|².
- **Prerequisite:** measuring time-resolved Fields over time partitions (array-backed rows, D96). It is moved to the top of the to-do list.

## 1. The four parts in nxr-neuro

| Part | nxr's name | nxr-neuro (where) | Exists? |
|---|---|---|---|
| **1. Loading** | `inputs` (found by the model, never by path) | `io`: the chunk service for recordings; a **finder** for stored nodes by kind, links and `where` | chunk service: yes; finder: **new** (port nxr's input resolution) |
| **2. Preparation** | the **plan** (`--dry-run`) | `pipelines`: inputs found, every arrow checked on the model, constants bound by `setup` steps | **new** |
| **3. The graph** | **steps** (registry, ports, kinds) | `compute`/`meeg`: each step a pure function; `pipelines`: the registry and the runner | functions: partly; registry and runner: **new** |
| **4. Readback → io** | **outputs** | `io`: each output stored as a Field on the manifold its steps made, with provenance | the writers: yes |

## 2. A definition (nxr's format, the same file in both tools)

```yaml
pipeline: band_power_source_map            # header: name, version, scope (subject)
version: 1
scope: subject
inputs:                                    # found by the model: a field / operator / selection of a kind, narrowed by links and `where`
  recording: {field: recording, each: true}
  kernel:    {operator: inverse kernel, from: recording.manifold.0}
  bad:       {selection: spans, manifold: recording.time, where: {name: "{recording}_bad_segments"}, required: false}
parameters:
  band: [8.0, 12.0]
steps:
  chunks:                                  # THE CHUNK LOOP: the chunk service hands each chunk (with the halo the steps ask for)
    over: recording
    do:
      - {step: band_filter,  in: {x: recording},             with: {band: $band}, out: x_band}
      - {step: apply_kernel, in: {x: x_band, kernel: kernel},                     out: sources}
      - {step: mean_square,  in: {x: sources, exclude: bad},                      out: power}
  after:
    - {step: orientation_sum, in: {x: power}, out: band_power}
outputs:
  band_power: {kind: band power, unit: "A²·m²"}
```

The rules are nxr's:
1. Five sections: the header, `inputs`, `parameters`, `steps` (`before` · `chunks` · `after`), `outputs`.
2. Inputs are found by the model, never by path.
3. Every name a step uses is an input or an earlier step's output (`in`, `out`, `with`, and `$parameter`).
4. A recording is read only in chunks; only `chunk` and `accumulate` steps go in the loop.
5. Only the `outputs` are stored.

**The span** (which chunks) is the caller's, on the command line (`--chunks K|A-B|all`, `--start/--stop`), as for every verb: chunk 0 by default.

## 3. Steps

A step is registered once with:
- its **kind**;
- its **ports**, each with a type;
- its **parameters** and their defaults;
- the **halo** it needs (chunk steps);
- the **reference** it is held equal to (a Studio CPU twin or an nxr step).

Its computation is a pure function in `compute/` or `meeg/`.

| Kind | Runs | Its output |
|---|---|---|
| `setup` | once, before the loop, never on recording data (e.g. binding a kernel, the mesh gradient) | used by any later step |
| `chunk` | on every chunk, with its halo (read with the chunk, dropped after) | stays in the loop |
| `accumulate` | `start · fold · merge · finish`: sums folded chunk by chunk (merge: workers' partial sums, in chunk order) | leaves the loop at the end |
| `after` | once, after the loop | |

**Port types** (what the plan checks):
- a **field** on a manifold (on M × time, on N, on a band) with a **value type**: ℝ scalar, ℂ complex, a vector3 / ℍ current, hermitian3, quaternion;
- an **operator** from one manifold to another;
- a **selection** (spans, a partition);
- a **constant** array.

A step makes its outputs on manifolds the plan knows (`apply_kernel` puts its output on the kernel's target), so an output needs no declared place.

**The first registry** (each step one PR, with its golden; the chain order is the Studio's):

| Step | Kind | In → out | Reference |
|---|---|---|---|
| `band_filter` | chunk | ℝ field on M × time → the same, band-limited (`timebank` gains on the bins: Meyer or Morlet; zero phase) | the Studio's `gainTaps` / `convolveRows`; nxr `band_filter` |
| `analytic` | chunk | ℝ → ℂ, the analytic signal (the transform "complex") | the Studio's odd taps (`hilbertImag`), `analyticOf` |
| `envelope` | chunk | ℝ → ℝ, \|analytic\| (non-linear: nothing linear may follow, D106) | the Studio's envelope |
| `apply_kernel` | chunk | field on M × time, operator M → N → field on N × time (ℝ or ℂ kept; a free kernel gives a vector3 / ℍ current) | nxr `apply_kernel` |
| `form` | chunk | a field of type T → a form of it (`x`, `nx`, `xt`, `div`, `curl` and variants, `absq`, …): the Studio's `FORMS[T]`, ℂ-extended (linear forms keep the phase; `abs`/`absq` give the envelope) | the Studio's `fieldForms.ts` (`formValues`) |
| `mean_square` | accumulate | → the mean of \|x\|² over the good samples | nxr `mean_square` |
| `orientation_sum` | after | vector3 → x² + y² + z² | nxr |
| `second_moments` | accumulate | ℂ vector field → ⟨J Jᴴ⟩ per vertex (hermitian3: the source tensor) | today's `sourcetensor` |
| `phase_sums` | accumulate (or chunk, per frame) | ℂ field (any number of components), mesh gradient → P, H, Q | nxr `core_phase` (`read_out`) |
| `phase_readout` | after (or chunk) | sums → gradient, velocity, speed, travelling index | today's `phaseReadout` |
| `congruence` | after | operator A, Hermitian M, a pattern → A M Aᴴ on the pattern | general algebra (the spectral route from a stored cross-spectrum) |

## 4. The runner (generic: no analysis-specific code)

1. **Plan** (`--dry-run` stops here):
   - inputs found (absent or ambiguous inputs refused, named);
   - every step's arrows checked on the model before any data is read: types, manifolds, channels against the kernel, `absq` refused as a phase
     input, D106;
   - the halo of the loop computed (the halos of the chain's chunk steps, composed);
   - every output's manifold known.
2. **Before:** the setup steps (constants bound once: the Studio's "uploaded once").
3. **The loop:** the chunk service's job (`chunk_loop`):
   - chunk k with the loop's halo, chunk k + 1 read meanwhile;
   - the chunk steps run on it, the accumulate steps fold it;
   - progress and cancel as today.
4. **After:** the after steps.
5. **Store:** each output a Field of the subject, on the manifold its steps made, named `<pipeline>.<output>.<anchor>` (nxr's naming, D157), with
   `producer_json`: the pipeline (name, version, the definition's sha256), the parameters, the inputs' ids, the chunks read, and nxr-neuro's version.
   The cache key is the same, so a rerun with the same definition, inputs and span is a cache hit.

The command:
```
nxr-neuro run DEFINITION --dataset D --subject S [--set name=value …] [--chunks K|A-B|all | --start S --stop S] [--dry-run]
```

## 5. Worked examples

**Band power map** (the definition above). It must equal today's `sourcepsd` and nxr's `band_power_source_map` on the same data: the first parity
test of the runner.

**Phase gradient, summary over a span:**
```yaml
inputs:  {recording: …, kernel: …, surface: {manifold: …, from: kernel.to}}
steps:
  before: [{step: mesh_gradient, in: {surface: surface}, out: G}]
  chunks: {over: recording, do: [
    {step: band_filter,  in: {x: recording}, with: {band: $band}, out: xb},
    {step: analytic,     in: {x: xb}, out: xa},
    {step: apply_kernel, in: {x: xa, kernel: kernel}, out: J},
    {step: form,         in: {x: J, surface: surface}, with: {form: $form}, out: Z},
    {step: phase_sums,   in: {z: Z, gradient: G}, out: sums}]}
  after:  [{step: phase_readout, in: {sums: sums}, with: {omega: $omega}, out: pg}]
outputs: {pg: {kind: phase gradient}}
```

**Phase gradient, per frame (a series):** the same steps with `phase_sums` and `phase_readout` as chunk steps, and a series output (see decision 2).

**The source tensor:** band_filter → analytic → apply_kernel → `second_moments` (accumulate). Its Stokes, normal-power and normal-share maps are
after steps.

**The spectral route:** from a stored sensor cross-spectrum, `congruence(kernel·form, C, edges)` → `phase_sums_from_moments` → readout, all after
steps, with no loop. Any Hermitian input can be named, not only the stored one.

## 6. Tests (scoped, through `tools/tests.py`)

- **Each step:** against its reference by a golden generated by the reference's own code (the Studio's CPU twin, or nxr's step), on synthetic data.
- **The runner:**
  - the plan refuses each bad arrow, with its reason;
  - a chunk read with its halo gives what the whole recording gives on its columns (nxr's `test_taps_parity`);
  - the result does not depend on the chunk width;
  - a definition run in nxr and in nxr-neuro gives the same numbers on a synthetic dataset.
- **Real-world** (by hand, not a ctest): a real-data recording (`NXR_FIXTURE_DATASET`), chunk 0, rendered by the Studio.

## 7. What changes in today's verbs

`sourcepsd`, `csd`, `sourcetensor` and `phasegradient` become **definitions shipped with nxr-neuro**, run by the runner. Their command names can stay as
shortcuts that run the shipped definition. Their hand-wired chains are removed. `meeg::phaseSums(A, B, S, Sq)` splits into `phase_sums`,
`phase_sums_from_moments`, `phase_readout` and `congruence`. The measurement verbs (`measure compute`, `measure export`) stay; they read the outputs.

## 8. Order of PRs (each scoped, each with its golden)

1. `form` (the Studio's forms, ℂ-extended).
2. `band_filter` and `analytic` (on `timebank`).
3. `phase_sums` (any number of components) and `phase_readout`.
4. The step registry and port types, and the plan (arrows checked, `--dry-run`).
5. The input finder (nxr's resolution) in io.
6. The runner (setup, the loop through `chunk_loop`, accumulators, after, store) and `nxr-neuro run`.
7. Parity: `band_power_source_map` against today's `sourcepsd` and nxr.
8. Today's verbs re-expressed as shipped definitions; their hand-wired code removed.
9. The real-world check on the fixture; rendered with the Studio.

## 8b. The rule: nxr's design as it is; only what C++ forces differs

The chain follows nxr's pipeline, with changes only where C++ needs them.

**The same as nxr, unchanged:**
- the definition's five sections;
- inputs found by the model;
- the four step kinds;
- **nxr's step names and ports** (new steps added to both tools);
- the plan and `--dry-run`;
- the runner's order;
- how outputs are named and stored.

**What C++ forces:**
1. **Reading the definition:** a YAML parser (decision 1).
2. **The registry:** a compile-time table, one entry per step (name, kind, typed ports, parameters and defaults, halo, reference, the function), where
   Python registers on the fly.
3. **The value on an edge:** one explicit type, a real or complex matrix with its manifold, value type (ℝ, ℂ, vector3/ℍ, hermitian3) and unit, where
   Python passes numpy arrays. The plan checks arrows with it before any data flows.
4. **Workers:** threads, with the accumulators merged in chunk order, where nxr uses processes. Without threads (WebAssembly) the runner works one chunk
   at a time.

## 9. Open decisions

1. **The definition format. Decided: nxr's YAML, read with rapidyaml** (a new `io` dependency; its dated amendment goes in `docs/reference.md`
   dependency table with the PR that adds it).
2. **Series output size (open: decided when the per-frame output is built).** A per-frame phase-gradient map of a 27 s chunk at about 300 frames/s on 20,484 vertices is about 2 GB. Choose a stride, or
   which readouts a series keeps.
3. ~~Shared registry names~~ **Decided: yes**, nxr's names and ports exactly, and new steps added to both (§8b).
4. **Workers. Decided: later.** One chunk at a time first.

## Checklist (the architecture policy)

- **Compute functions callable with any matrix:** yes. Every step is a pure function, tested alone.
- **The same chain on another input of the same kind, by naming it:** yes. Inputs are found by the model from the definition.
- **Every loop and file access in io:** yes. The loop is the chunk service's; stores are io's.
- **Every choice of which data made by the caller:** yes. The definition and the command line choose; no step picks its input.
