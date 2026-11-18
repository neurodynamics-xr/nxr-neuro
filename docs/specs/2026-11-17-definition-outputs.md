# Definition outputs: every kind of result a definition stores, the accumulate nodes of spectra, presets (spec, 2026-11-17)

Status: **design, for the maintainers' review before code.** It completes `2026-11-17-compute-and-run.md` (the runner's fourth job: store) and
`2026-11-12-chain-graph.md` §2 (the `outputs:` section). It is the foundation on which the hand-wired `<x>_compute` verbs become definitions
(`ARCHITECTURE.md`, "Decided structure", 3). `psd` is converted on it first, as the proof.

## 1. Why

A definition can store only what the runner knows how to store. Today the runner stores a Field, measurement rows, a spans or set Selection, and a
statistic through five ad hoc writers chosen by keys in an array's `meta` (`generic_statistic`, `histogram`, `baseline`, `home_map`, summaries).
None of them can write what a spectrum needs (a channel axis on the recording's channels, a frequency line, the segment partition, the scaling
record), and none writes an Operator (a kernel, a head model, eigenmodes) or a partition. So about 38 verbs stay hand-wired: each finds its inputs,
loops and stores by itself. This spec makes storing **one generic mechanism**: every model object a verb produces is an output **kind** declared in
the definition, described by the node that computes it, and written by io in the run's one `task::Unit`.

## 2. The rule: the node describes, the definition places, io writes

| who | says | knows |
|---|---|---|
| **the node** (pure, `compute/` + its registration) | what the values ARE: measure, method, axes (sizes, named entries, a closed-form line's `n`, `spacing`, `origin`, unit), the partition it reduced over as a tiling (origin sample, width, hop, count, members skipped), the scaling record, the unit, the options it resolved | arrays and parameters; never an id, a name or a path |
| **the definition** (`outputs:`) | WHERE and OF WHAT: the kind, the stored name, the found input it is of, which found manifold each axis lies on, the names of the line and the partition it makes | the inputs it found (by link: `recording.manifold.0`, `recording.time`, `kernel.to`) |
| **the runner** | binds the two: links resolved to the found nodes, `{name}` templates filled, the description checked against the declaration (shape against axes) | the plan; no step logic, no per-kind knowledge beyond this table |
| **io** (`task::Unit`) | writes rows, arrays, the made line and partition, provenance (`By`: the producer record with the definition, its sha256, parameters, inputs, span) and the cache key, atomically | rows and SQL (only io) |

The node's description travels in the output array's `meta.describe` (a JSON object; the keys per kind in section 3). The plan checks a declared
output against its node's output port type before any data (kind, rows, value type); the store checks the shape against the axes.

## 3. The kinds

Every output is `{kind: K, name: TEMPLATE, …}`. `name` defaults to nxr's `<pipeline>.<output>.<anchor>` (as today); a template may use the
parameters and the found inputs' names (`"{recording}_psd_welch"`, `$name`). The existing short forms stay valid, unchanged: `{kind: spans}` is
`{kind: selection, type: spans}`, `{kind: labelling}` is `{kind: selection, type: set}`; the shipped definitions need no edit.

| kind | declared (YAML) | described by the node (`meta.describe`) | io writes (`Unit`) | verbs it serves |
|---|---|---|---|---|
| `field` | `on: LINK` (optional: the plan's manifold otherwise), `kind_name` (the field's kind) | value type and unit (from the port type) | a Field on that manifold | today's field outputs; `frames`, `sourcepsd` maps |
| `statistic` | `of: INPUT`, `axes: [AXIS…]`, `partition: PART` (optional) | `measure`, `method`, `axes` [{name, size, values \| edges \| line}], `tiling`, `n_members`, `members_skipped`, `components`, `scaling`, `options`, `unit`, `description` | the statistic, the line(s) and the partition it names | `psd`, `crossspectrum`, `statistic`, `pyramid reduce`, today's five statistic writers |
| `operator` | `operator_kind`, `from: LINK`, `to: LINK` | layout (dense, csc), rows and cols, `params`, unit | an Operator from → to | `inverse`, `headmodel`, `noisecov`, `eigenmodes` |
| `selection` | `type: partition \| spans \| set \| tiling \| ladder`, `on: LINK` | codes per element (partition, set), spans [{start, stop, code}], or a closed-form `tiling`; dictionary entries; `params` | a Selection on that manifold (and its dictionary) | `tiles`, `patches`, frame tiling, detections |
| `measurement` | `over: INPUT \| OUTPUT` (a partition), `of: INPUT` | per member: `measure`, value(s), count, unit | measurement rows on that partition | `measure`, `cellquadrature`, today's measurement outputs |
| `manifold` (amended 2026-11-17) | `type: line \| surface \| product`; a line: `{name: TEMPLATE, alt: TEMPLATE}`; a surface: `cs: KEY`, `derived_from: LINK`; a product: `factors: [LINK \| OUTPUT…]` | a line: `start`, `step`, `count`, `unit`, `dimension`; a surface: vertices [n × 3], faces [m × 3] | the manifold, found by name and reused when identical, never replaced | time, frame, frequency and mode lines; `headmodel` and template surfaces; sensor × time products |

**Manifolds first (amended 2026-11-17, the maintainers).** A manifold is the sixth kind. Any other output of the same definition may lie on a manifold the
definition makes (`on: OUTPUT`, an axis `on: OUTPUT`, a product's factor); the runner writes the manifolds first, in the same `task::Unit`, then the rest.
A statistic's `line:` axis below is this kind written inline (the same `Unit::line`), not a mechanism of its own.

An `AXIS` of a statistic is `{name, on: LINK}` (its entries are the elements of a found manifold: the recording's channels) or `{name, line: TEMPLATE}`
(a closed-form line made from the node's `line` description, found by name and reused when it fits, else named `TEMPLATE_alt`: today's
`frequency_line`), or `{name}` alone (named entries or edges the node gives). A `PART` is `{tiling: TEMPLATE, on: LINK}`: the node's tiling stored as a
closed-form spans Selection on a time line (today's `welch_segments`) and named as the statistic's partition, its members the segments.

## 4. io: the `Unit` methods

The task API gains typed writes on `task::Unit` (`io/include/nxr/io/task/task.h`); each takes task-level structs that name found nodes by id and
arrays as views, never a column. Inside io they call `SubjectWriter` as today.

```cpp
std::string field(const FieldOut&);          // name, manifold_id, kind, value_type, unit, data
std::string statistic(const StatisticOut&);  // name, of (field or statistic id), measure, method, axes (JSON: manifold ids bound), partition id, span, scaling, options, unit, data
std::string operator_(const OperatorOut&);   // name, kind, from/to manifold ids, layout, rows, cols, params, data (dense or sparse parts)
std::string selection(const SelectionOut&);  // name, type, manifold_id, codes | spans | tiling, dictionary entries, params
std::string line(const LineOut&);            // a closed-form line (n, spacing, origin, unit, folder): found by name and reused when it fits, else made
std::string surface(const SurfaceOut&);      // vertices and faces from a node's arrays, its coordinate system, the surface it derives from; reused when identical
std::string product(const ProductOut&);      // a product of existing factors or factors made in the same unit
std::int64_t write_measurements(const Json&);   // exists
std::optional<std::string> replace(Cls, const std::string& path);   // --replace: the node at that path moved aside (never deleted)
```

Every node written takes the Unit's `By` (producer record and cache key) with the output's name added, as today. `pipelines/src/cache.cpp`'s
`clear_stored` becomes `Unit::replace`. The runner's five statistic writers become one `store_output` per kind; the `meta` keys they read are rewritten
as `describe` in the nodes that set them, in the PR that converts each node's pipeline (the layering allowlist only shrinks).

## 5. Accumulate nodes of spectra, pure

- **`signal.psd`** (accumulate; stateful in time: segments are carried across blocks, so blocks meet in order). In: `x` (real, samples), optional
  `exclude`. Out: `psd` [rows × bins], float64, described as a statistic (measure `power_density` or `power`, method `welch`, axes channel and
  frequency `line`, the segments' `tiling`, the scaling record). Parameters, each with meaning, default and unit: `segment_seconds` (4, s),
  `segment_samples`, `segment_level`, `segment_rule` (`tile` | `round` | `even`), `overlap` (0.5), `window` (hamming), `periodic` (false), `detrend`
  (constant), `scaling` (density), `bad_segments` (exclude). The maths is `nxr::welch::Welch` unchanged; the segment rule (today's `welch_segment`)
  becomes a pure `nxr::welch::segmentFor(rule, …)` in `compute/`.
- **`signal.linear_map`** (chunk, pointwise, a linear map): z = a_x·X + i a_y·X from a stored [2 × C] operator. **`signal.complex_psd`** (accumulate): the
  two-sided Welch of z on a signed frequency line.
- **The span's grid.** Under `segment_rule: tile` a segment is 2^k samples on the line's grid, so the span snaps to level k. A node states it in its
  plan: `Planned::align` (samples); `check` takes the largest over the graph; the runner hands it to the chunk service's span (today's `level`).
- **Shared with `sourcepsd`, `crossspectrum`, `sourcetensor`:** the segment rule (`segmentFor`), the scaling record (`welch_scaling_record`, moved
  beside the node registrations: it needs no io), the frequency line and segment partition (`Unit::line`, `Unit::selection` with a tiling). Their
  private copies in `meeg.cpp` stay until each verb is converted.

## 6. Presets: named parameter sets

```yaml
presets:                                    # each a named alternative (the method rule: alternatives are named options); the defaults are the parameters'
  tiles:      {segment_rule: tile}          # the default, named so that it can be asked for
  brainstorm: {segment_rule: round, window: hamming, periodic: false, detrend: constant, scaling: density}
  scipy:      {segment_rule: round, window: hann,    periodic: true,  detrend: constant, scaling: density}
  nxr-chain:  {segment_rule: even,  window: hann,    periodic: true,  detrend: none,     scaling: chain}
```

- `run psd --set brainstorm` selects one (a `--set` value without `=`); the caller's own `--<parameter> VALUE` then wins over it: defaults < preset
  < the caller's. A preset names only declared parameters (refused otherwise, when the definition is read).
- The preset's name goes into the provenance and the cache key (`preset`), beside the resolved parameters.
- `run <pipeline> --help` lists every parameter (meaning, default, unit) and every preset with the values it sets.

## 7. `psd` as a definition (the proof)

```yaml
pipeline: psd
inputs:
  recording: {field: recording, where: {name: $recording}}
  bad:       {selection: spans, manifold: recording.time, where: {name: "{recording}_bad_segments"}, required: false}
parameters: {recording, name: "{recording}_psd_welch", segment_seconds, …, bad_segments}   # long form: meaning, default, unit each
presets:    {tiles, brainstorm, scipy, nxr-chain}
steps:
  chunks: {over: recording, do: [{node: signal.psd, in: {x: recording, exclude: bad}, with: {…: $…}, out: psd}]}
outputs:
  psd:
    kind: statistic
    name: $name
    of: recording
    axes: [{name: channel, on: recording.manifold.0}, {name: frequency, line: "{recording}_freq"}]
    partition: {tiling: "$name_segments", on: recording.time}
```

The rotary spectrum is its own definition, `rotary_psd` (`--operator NAME`: `signal.linear_map` then `signal.complex_psd`; axes pair × a signed line): a
different graph with a different output, not a mode of `psd` (question 1). Stored names, paths and the line are today's (`<recording>_psd_welch`,
`<name>_segments`, `<recording>_freq`), so every reader (`vortex_null`, `spatial_events_null`, `wishart`) is unchanged. `psd_compute` and its
command-table entry go. `--no-bad-segments` becomes `--bad-segments ignore`; `--periodic`/`--symmetric` become `--periodic true|false`.

**Tests** compare with the REFERENCES, not with the old output (decided): the run on the synthetic signal of `compute/test/golden/welch_golden.txt`
equals the scipy and Brainstorm goldens per preset; the rotary run equals `rotary_welch_golden.txt`; the result does not depend on the chunk size;
the stored record (axes, line, partition, scaling, unit, provenance, cache hit on rerun) is checked. `test_psd_pipeline`, `test_meeg_pipelines` and
`test_source_psd` are rewritten so; tests that only used `psd_compute` to make a fixture call the runner.

## 8. PRs

| # | contents |
|---|---|
| D0 | this spec |
| D1 | io: the `Unit` methods of section 4 and their structs (the six kinds, manifolds included), tested on synthetic outputs of every kind; the runner's `store_output` per kind from the declaration and `describe`; the definition reader accepts the declarations (short forms kept); a toy definition per kind in the runner's tests |
| D2 | `signal.psd`, `signal.linear_map`, `signal.complex_psd` registered in `nxr::graph::registry()` (grouped names; a definition says `node:`); `Planned::align`; presets in the definition reader, `run` and `--help` |
| D3 | `psd.yaml` and `rotary_psd.yaml`; `psd_compute` deleted; the tests of section 7; `docs/reference.md`; `CONTRIBUTING.md`: how a verb is converted (outputs, nodes, presets, tests) |

## 9. Questions, each with a recommendation

| # | question | recommendation |
|---|---|---|
| 1 | `--through`: a parameter of `psd`, or its own definition | **its own definition, `rotary_psd`**: another graph and another output; a definition has no conditional steps, and a node must not switch on an optional input |
| 2 | PSDs stored by the hand-wired verb | readable as before; not cache hits of the definition (another key): a rerun under the same name needs `--replace` |
| 3 | Where the new nodes are registered | `nxr::graph::registry()` with grouped names (C6's direction), beside the runner's steps; the runner looks a use up in both |
| 4 | Today's five statistic writers | kept until each node's pipeline is converted, then rewritten as `describe`; never extended |

## Checklist (the architecture policy)

- **Each new compute function callable with a matrix from anywhere?** Yes: `segmentFor` and the nodes take arrays and parameters; a node's description
  holds no id.
- **The same chain on another input by naming it?** Yes: outputs are placed by links to found inputs, never by path.
- **Every loop over samples or chunks, and every file access, inside io?** Yes: the chunk service streams; `Unit` writes.
- **Every choice of data made by the caller?** Yes: inputs, span, preset and parameters are the caller's; presets only name parameter values.
