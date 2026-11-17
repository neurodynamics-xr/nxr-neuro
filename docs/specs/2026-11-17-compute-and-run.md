# `compute` and `run`: nodes and graphs on data handed to them, pipelines as io around the same executor (spec, 2026-11-17)

Status: **design, no code.** The maintainers decided the model on 2026-11-17; this spec turns it into interfaces, a grammar, the bindings' API and an
ordered plan of PRs. It amends `2026-11-12-chain-graph.md` (the runner, the registry), `2026-11-16-command-line.md` (the `compute` and `run` verbs) and
`2026-11-17-interoperability.md` (the bindings' levels), and it is sequenced with the io record API migration (`2026-11-17-io-record-api.md`, section 0).
Open questions are in section 10, each with a recommendation.

## 1. Why

nxr-neuro measures the spatiotemporal patterns of cortical activity. A measurement is worth something only if the same computation can be checked on
any data: a synthetic array in a test, a file from another tool, a stored recording streamed chunk by chunk. Today these are three code paths:
- the registered steps run only inside the runner, on a subject, through the database;
- the pure functions are bound by hand in Python (`nn.ops`) and MATLAB (`nxr.ops`), one by one, with no link to the steps a pipeline uses;
- every `compute <what>` command on the command line is a pipeline on a dataset, so no computation can be applied to a file or an array.

The architecture already says the pipeline is a GRAPH of pure functions (`ARCHITECTURE.md`, part 3). A graph of pure functions is itself a pure function,
so graphs belong at the pure level, beside single nodes. This spec draws that line through the code:

| verb | what it does | knows about |
|---|---|---|
| **`compute <node or graph>`** | applies a node, or a graph of nodes, to data handed to it | arrays (and, on the command line, files named by the caller); never a dataset, a subject, a finder, a cache |
| **`run <pipeline>`** | runs a pipeline: io around the same executor | finds the inputs, streams the stated time range through the chunk service, stores the outputs with provenance and cache key |

## 2. The model

**Node.** One pure function of matrices with typed input and output ports, parameters with defaults, declared traits (time behaviour, precision), and the
reference it is held equal to. It is registered once (the registry), under a **grouped name** (`welch.psd`, `filter.bandpass`, `band.analytic`). A node has
one of three kinds:

| kind | runs | its inputs → outputs |
|---|---|---|
| `array` | once, on whole arrays | arrays → arrays (today's `setup` and `after`, which differ only in where a definition puts them) |
| `chunk` | on every block (a chunk and its halo, or a whole array handed as one block) | blocks → blocks; pointwise or windowed; a `stateful` chunk node carries state |
| `accumulate` | `start · step · finish` over the blocks in time order | blocks → arrays; a reduction over a time partition (the cardinal rule: a measurement over a partition, or a statistic of the partition's members) |

**Graph.** Nodes wired port to port: named inputs, parameters, uses of nodes in three sections (`before` · `chunks` · `after`), named outputs. It is
the `steps:` part of today's definition, without the inputs' finding and the outputs' storing.

**Graph as node (a composite).** A graph wrapped with its own ports and parameters becomes a NEW NODE: registered under a grouped name, listed, callable by
`compute`, usable in any other graph. Its kind, traits and halo are derived from its inner nodes (section 4). This is TSL's `Fn` (section 5).

**Executor (pure level).** Checks a graph (arrows, ports, kinds, traits, halo) before any data, then runs it: `before` once, each block through the chunk
and accumulate nodes with the state carried across blocks ((block, state) → (output, state)), `after` once. It knows nothing of files, chunks' origin,
subjects or storage. It is handed blocks one at a time; it never loops over a recording.

**Runner (orchestration).** `run <pipeline>` and nothing else:
1. **find** the inputs (io's task API: `task::Subject::find`, `members`, `spans`);
2. **bind** what was found as typed arrays (stored constants read by name; the recording's port type: rows, rate, unit, value type);
3. **stream** the caller's time range through io's chunk service (`haloed_loop`) with the halo the checked graph declares, handing each chunk to the executor;
4. **store** the outputs with provenance and cache key (one unit, `Subject::commit`).

The runner keeps no step logic. **Built-in pipelines** are of two forms, both run by `run`:
- the hand-wired C++ verbs (today's `compute psd`, `compute sourcepsd`, …), dispatched as they are: `run psd` calls `psd_compute` with the options it
  parses today. Converting them into YAML definitions is NOT part of this spec (a separate decision for the maintainers);
- YAML definitions, shipped with nxr-neuro or the user's own (a path).

**Naming.** Pipelines have plain names (`psd`, `global_bursts`): no dot. Nodes have grouped names (`welch.psd`, `filter.bandpass`): `<group>.<name>`,
with a dot. They never collide; the registry and the pipeline table each refuse a name of the other form, and a test checks both tables.

**Reductions stay where the architecture puts them.** A node is pointwise or an `accumulate` node that reduces over a TIME partition handed to it; ratios are
derived measures computed at read. In `run`, measurements and statistics are stored and read back through io. In `compute`, a partition's memberships are
handed to the graph as arrays (an integer member index per sample or per vertex, `-1` for none), and the output is the measurement's ingredients per
member (the weighted sums and their weight), exactly what `run` stores.

## 3. The executor's interface (C++)

**Where.** A new library, `nxr_graph`, in `pipelines/graph/` (headers `pipelines/graph/include/nxr/graph/*.h`, namespace `nxr::graph`). It depends on
`compute`, `meeg`, Eigen and nlohmann/json, **never `io`** (a layering rule, enforced as the recordings rule is). `nxr_pipelines` links it; the bindings link
it without io (so the WebAssembly `compute` build gets the graph). The dependency table gains one row (question 1).

### 3.1 Values and types

```cpp
namespace nxr::graph {
using Json = nlohmann::json;

enum class ValueType { real, complex, vector3, quaternion, hermitian3 };
enum class PortClass { field, operator_, selection, partition, constant };
enum class Axis { none, samples, frames, members };       // what a field's columns are

// What a port carries, known before any data: the plan checks arrows on these. No database, no row: the runner builds one from a found node,
// `compute` from the array handed to it and the caller's metadata.
struct PortType {
  PortClass cls = PortClass::field;
  ValueType value = ValueType::real;
  std::int64_t rows = 0;                 // elements (channels, vertices, members), a vertex's components consecutive
  Axis axis = Axis::none;                // samples: streamed (inside the loop); frames: retimed; none: an array
  std::optional<double> sfreq;           // Hz, for samples (frames: the frame rate)
  std::string unit;                      // SI, as stored
  std::string on, to;                    // the manifold an element lies on (a label the caller gives: "sensors", "cortex"); an operator's target
  bool nonlinear = false;                // made by a non-linear node: a linear map may not follow (D106)
  Json facts = Json::object();           // what a node's plan may read beside the shape: a ladder's tiling, a selection's parameters, a channel list
};
using Types = std::map<std::string, PortType>;

// Values: today's Block (inside the loop: float32, row-major, its halo round it) and Array (outside: double), moved here unchanged.
struct Block { std::int64_t rows = 0, width = 0, left = 0, right = 0; std::vector<float> re, im; /* own(), complex() */ };
struct Array { std::int64_t rows = 0, cols = 1, comps = 1; std::vector<double> data; Json meta = Json::object(); };
using Blocks = std::map<std::string, Block>;
using Arrays = std::map<std::string, Array>;
struct Place { std::int64_t first = 0, n = 0; const std::vector<std::uint8_t>* good = nullptr; };   // the block's own columns and their good mask
}
```

`NodeTraits` (time behaviour, precision, `window_rule`, `uncertain`, `note`) moves from `registry.h` to `nxr/graph/node.h` unchanged.

### 3.2 A node

```cpp
struct PortSpec  { std::string name, meaning; bool optional = false; };
struct ParamSpec { std::string name; Json def; std::string meaning, unit; std::vector<std::string> choices; };   // help is made from these

struct NodeSpec {
  std::string name;                       // grouped: "welch.psd"
  Kind kind;                              // array · chunk · accumulate
  std::vector<PortSpec> inputs, outputs;
  std::vector<ParamSpec> parameters;
  NodeTraits traits;
  std::string matches, description;       // the reference it is held equal to; one sentence

  // PLAN: the arrows, on types only (no database: every fact a node needs is in its inputs' PortType). Returns the outputs' types and the halo
  // it needs on each side (a chunk node; 0 otherwise), or refuses (refuse("...") naming the port and the reason).
  std::function<Planned(const Types& in, const Json& params)> plan;

  // PREPARE: once, before any block: the constants bound (taps designed, a kernel's matrix, the mesh gradient), from the parameters and the array
  // inputs that are constants. Its result is immutable for the run (the Studio's "uploaded once").
  std::function<std::any(const Types& in, const Arrays& constants, const Json& params, Shared& shared)> prepare;

  // chunk, pointwise or windowed: a block in, a block out, `keep` samples of halo left on each side for the nodes after it
  std::function<Blocks(const std::any& prepared, const Blocks& in, std::int64_t keep, const Place& at)> chunk;

  // stateful chunk and accumulate: (block, state) -> (output, state). State is a value, moved in and moved out; the executor owns it.
  std::function<std::any(const std::any& prepared)> start;
  std::function<std::pair<Blocks, std::any>(const std::any& prepared, std::any state, const Blocks& in, std::int64_t keep, const Place& at)> step;
  std::function<Arrays(const std::any& prepared, std::any state)> finish;      // accumulate: the state's reduction as arrays

  // array: arrays in, arrays out
  std::function<Arrays(const std::any& prepared, const Arrays& in)> run;

  // a retiming chunk node (the frame tiling): where a block's own columns fall as frames
  std::function<FramePlace(const std::any& prepared, const Place& samples)> frame_place;
  // what preparation resolved that the parameters leave open (a series degree, a transform's error): into the provenance
  std::function<Json(const std::any& prepared)> record;
};
```

Differences from today's `StepSpec`:
- `plan` takes `PortType`s, not `finder::Value` with a `PlanContext` holding the database. A fact a step looked up in the database (a glued surface's id, a
  ladder's level) becomes a fact of its input's type, or a declared input of its own, bound by the runner (PR C4).
- `prepare` takes the constants as arrays, not an `ExecContext` holding the subject. A node never reads or stores a node by name; the runner does
  (PR C4 lists the steps that do today, including the spatial band's "compute and store the eigenmodes when absent").
- `resolve` (a request resolved on what is stored, part of the cache key) leaves the node: it is the runner's binding step, whose result is part of the
  bound input's type (`facts`) and so of the key.
- `fold(state&, …)` becomes `step(prepared, state) -> (output, state)`: the same work, the state explicit and owned by the executor.
- `halo` is part of `plan` (known before any data), not of the prepared state.

### 3.3 Checking a graph

```cpp
struct Graph {                                       // the steps of a definition, or the body of a composite (section 4)
  std::string name; int version = 1; std::string text;   // its canonical text; sha256(text) goes into the provenance and the cache key
  std::vector<std::string> inputs, outputs;
  Json parameters;                                   // name -> {default, meaning, unit}
  std::vector<Use> before, chunks, after;            // Use: {node, in: port -> value name, with: param -> value | $name, out: names}
};

struct Checked;                                      // immutable: the flat node list, each node's types, keep and prepared-state slot; the loop's halo; the outputs' types

/** every arrow checked before any data: refuses (invalid_argument) with the use, the port and the reason */
Checked check(const Graph& g, const Types& inputs, const Json& params, const Registry& registry = registry());
```

`check` does, in order:
1. **Structure:** sections known; every name used is an input or an earlier output; every `$name` a parameter; parameters' types match their defaults.
2. **Nodes:** each `node:` found in the registry (an unknown name is refused with the near names); a composite is **expanded** in place (section 4.3);
   a composite that reaches itself is refused (the cycle named).
3. **Traits:** every node declares them (`check_traits`); a `stateful` node is refused in a block-parallel context.
4. **Kinds by section:** `chunk` and `accumulate` nodes only in `chunks`; a `before` node takes no streamed value; an `after` node takes no block.
5. **Arrows:** each node's `plan` on its inputs' types, in order: the outputs' types, or a refusal. The checks today's plan makes stay, now on types:
   value types (ℝ/ℂ/vector3/ℍ/hermitian3), rows against an operator's columns, `on`/`to` manifolds, units, `absq` refused as a phase input, D106
   (`nonlinear` carried forward, a linear map after it refused).
6. **Halo:** composed backwards over the chunk nodes. With h(n) the halo node n needs and keep(n) the halo it must leave for its consumers,
   keep(n) = max over the chunk nodes c reading n of (h(c) + keep(c)), 0 for a node read only by accumulate nodes or outputs; the loop's halo is
   max over the nodes reading a streamed input of (h(n) + keep(n)). (Today's runner computes the same; it moves here.)
7. **Retiming:** the nodes after a retiming node (the frame tiling) run on frames; their `Place` is the frame's.
8. **Outputs:** every output is a value; its type known. Streamed outputs (axis samples or frames) and array outputs are both allowed.

`Checked::text()` is the plan as a person reads it (`--dry-run`), as `Plan::text` today, without row names (the runner adds them).

### 3.4 Running a checked graph

```cpp
class Execution {
 public:
  Execution(const Checked& c, const Arrays& constants, const DispatchContext& ctx = {});   // prepare every node; run `before`
  std::int64_t halo() const;                       // what each block handed must carry on each side
  // ONE BLOCK: the streamed inputs on [first − halo, first + n + halo), its place and good mask. Runs the chunk nodes (in sub-blocks of
  // `block_samples` after the last windowed node; frames in blocks of `block_frames` after a retiming node) and steps the stateful and accumulate
  // nodes in time order. Returns the streamed outputs' own columns.
  Blocks block(const Blocks& in, const Place& at);
  Arrays finish();                                 // finish the accumulate nodes; run `after`; the array outputs
  Json record() const;                             // every node's resolved options (record), for the provenance
};

/** convenience: whole arrays handed as ONE block (halo reflected at both ends, as at a recording's ends): the streamed outputs and the array outputs,
 *  as arrays; the same as `block` + `finish` */
Arrays apply(const Checked& c, const Arrays& inputs, const Arrays& constants, const DispatchContext& ctx = {});
```

- **Pure in form.** The execution is the function (blockₖ, Sₖ) → (outputₖ, Sₖ₊₁), S the states of the stateful and accumulate nodes; `Execution` holds S
  between calls so callers need not. A block's values never depend on how the input was cut into blocks (tested, section 9).
- **The executor does not loop over a recording.** It is handed blocks. In `run`, io's chunk service calls it once per chunk; in `compute`, the caller
  hands whole arrays (`apply`), or a file reader streams blocks (section 6.1). The sub-block and frame-block loops are inside one block it was handed:
  a memory bound on the pointwise tail, as inside any compute function.
- **One dispatch point.** Every node call goes through `dispatch(node, NodeCall, context)` (`dispatch.h` moves to `nxr/graph/dispatch.h`, unchanged in
  role); the guard `dispatch_is_the_only_call_path` follows it.
- **The node traits describe streaming.** `pointwise` and `windowed` nodes may be given blocks of any width and order (a windowed one with its halo);
  `stateful` and `accumulate` nodes are given blocks in time order. A future block-parallel backend reads this, nothing else.

## 4. Composite nodes: a graph wrapped as a node

### 4.1 The YAML

A composite is a graph with **ports instead of found inputs, and output ports instead of stored outputs**. It follows the definition format (the same
sections, `in`/`with`/`out`, `$parameter`) with three differences: the header says `node:` (a grouped name), `inputs` are typed ports, and `outputs` name
the values they expose.

```yaml
node: phase.current_of_band            # a grouped name; refused if a built-in node or a pipeline has it
version: 1
description: the phase current of one band on the cortex, from sensor signals
matches: the Studio's chain (members → filter → transform) with fieldForms.ts and the phase current
inputs:                                # PORTS: what a caller (compute, or a graph using this node) hands
  x:       {class: field, value: real, axis: samples, meaning: "sensor signals [channels x samples]"}
  kernel:  {class: operator, meaning: "the inverse kernel [sources x channels]"}
  surface: {class: constant, meaning: "the cortical surface's vertices and faces"}
parameters:                            # each with its default, meaning and unit: help and the bindings' signatures are made from these
  band:  {default: null, meaning: "the band's edges", unit: Hz}
  shape: {default: morlet, meaning: "the band's shape", choices: [morlet, meyer]}
  form:  {default: x, meaning: "the form of the current analysed (fieldForms)", choices: [x, nx, xt, div, curl]}
graph:
  before:
    - {node: mesh.gradient,  in: {surface: surface},            out: G}
  chunks:
    - {node: band.analytic,  in: {x: x},           with: {band: $band, shape: $shape}, out: xa}
    - {node: kernel.apply,   in: {x: xa, kernel: kernel},                               out: J}
    - {node: field.form,     in: {x: J, surface: surface},      with: {form: $form},   out: Z}
    - {node: phase.current,  in: {z: Z, gradient: G},                                   out: j}
    - {node: field.power,    in: {x: Z},                                                out: p}
outputs:                               # OUTPUT PORTS: values of the graph, with what they are
  current: {from: j, meaning: "the phase current Im(z̄∇z) per vertex per sample", unit: "A²·m"}
  power:   {from: p, meaning: "|z|² per vertex per sample", unit: "A²·m²"}
```

Shipped composites live in `pipelines/graph/nodes/*.yaml` (compiled in, as the shipped definitions are); a user's are found on a search path (question 5).
A pipeline definition uses a composite as any node (`{node: phase.current_of_band, …}`); so does another composite.

### 4.2 How a graph becomes a node

`NodeSpec wrap(const Graph& g)` makes the registry entry:
- **name, ports, parameters** from the YAML; port and parameter names are distinct (refused otherwise, so the command line can name both);
- **kind**, derived: no streamed input → `array`; streamed inputs and only streamed outputs → `chunk`; streamed inputs and an array output → `accumulate`
  (mixed outputs are allowed: section 10, question 4);
- **traits**, derived: time behaviour the strongest of its inner nodes (stateful > windowed > pointwise), `window_rule` the composed halo in words;
  precision `float64` if any inner node computes in it, output `float64` if any output port is made by a `float64` node; `uncertain` if any inner node is;
- **plan**: `check` of the body on the caller's types and parameters (so the arrows inside are checked with the real types at every use);
- **halo**: the body's loop halo;
- the execution functions are never called: the composite is expanded before execution (4.3).

### 4.3 Expansion (TSL inlines; so does the executor)

`check` expands every composite use in place before the arrows are checked: each inner use is renamed `<use>/<inner name>`, its ports wired to the
caller's values, `$parameters` replaced by the values the caller gave (or the composite's defaults). The executor sees ONE flat graph: the halo is composed
once over all nodes, every node goes through the one dispatch point, and a later GPU backend can fuse across the boundary. The provenance keeps the
structure: `record()` lists the composite (name, version, the sha256 of its text) and the expanded graph's text, so a result names what the user wrote.

## 5. The Studio's TSL chains: what is followed, what differs

The Studio's GPU passes are written in three.js's TSL (`studio/src/studio/engine/gpu/bank.ts`, `atomFilter.ts`,
`studio/src/engine/compute/pipelines/chebyshevGpu.ts`). What they show, and how this spec follows it:

| TSL, as the Studio uses it | nxr-neuro |
|---|---|
| `Fn(() => …)` composes node expressions into a reusable function; a helper such as `chebyshevGpu.ts`'s `sAt(cur, i)` is itself a node-making function used inside others | a graph wrapped as a composite node (section 4), used inside other graphs |
| calling a `Fn` inlines its body into the caller's node graph when the kernel is built | composites are expanded at `check` (4.3) |
| a `Fn`'s signature can be declared (`setLayout`: name, inputs and their types) | a composite declares typed ports and parameters |
| types are known per node when the kernel is built; a mismatch fails the build, not the run | every arrow is checked on `PortType`s before any data |
| a kernel is built once and cached by a signature of its shapes (`bank.ts`'s `sig`); per call only `uniform`s and buffers are written | `check` once per (graph, input types, parameters); `prepare` binds constants once; per chunk only blocks change |
| a resident operator is uploaded once and shared by a content key (`chebyshevGpu.ts`'s `hold`) | constants bound in `prepare`, shared by key between nodes (`Shared`, today's `ExecContext::shared`) |
| `.compute(N)`: one dispatch over N threads | `dispatch(node, NodeCall)`: one call per node per block |

Where C++ and this design differ, and why:
1. **Granularity.** A TSL node is an expression on one element (a thread); an nxr-neuro node is a function of matrices on a block. Composition is of compiled
   functions, not of code: there is no code generation, and a composite is a graph of registered nodes, never new arithmetic.
2. **State is explicit.** A TSL dispatch carries no state; the Studio keeps accumulators on the host between passes. The executor owns the state of stateful
   and accumulate nodes as values ((block, state) → (output, state)).
3. **Types are runtime descriptors.** TSL infers WGSL types from the expression; nxr-neuro's `PortType` adds what a measurement needs and a shader type has no
   word for: the manifold, the axis, the rate, the unit, D106's non-linearity.
4. **Parameters are values, not nodes.** `seedNode(value)` in `chebyshevGpu.ts` takes a node-making function as an argument; a composite's parameters are
   values only (a node passed as a parameter is question 9).

## 6. The command line

### 6.1 `compute <node or graph>`: files in, files out

```
nxr-neuro compute <node | composite | graph.yaml>  <input>... <output>...  [--<port> FILE]... [--<parameter> VALUE]...
                  [--sfreq HZ] [--unit PORT=UNIT]... [--from S --to S | --all] [--replace] [--json]
nxr-neuro compute <node> --help
nxr-neuro list nodes [GROUP]
```

- **Ports.** The required ports are positional, inputs then outputs, in the node's declared order (as `wb_command`, MRtrix and FSL take them); an optional
  port is `--<port> FILE`. A file names one array, with a selector when the file holds several: `x.mat#var`, `maps.func.gii#2` (a data array), `run.edf`
  (a recording: every channel, or `#MEG` for a channel kind).
- **Parameters** are `--<parameter> VALUE`, generated from the node's `ParamSpec`s (VALUE read as YAML: `8`, `[8,12]`, `meyer`). `--help` prints the ports and
  every parameter with its meaning, default and unit, and the node's traits and reference. A graph file's `--help` is made from its own declarations.
- **Formats** (through `io/formats`, the pure readers and writers of the interoperability spec): in, NumPy `.npy`, MATLAB `.mat` (v5 and v7.3), GIFTI,
  NIfTI, and the recording formats `io/formats` reads (EDF/BDF, BrainVision, EEGLAB, FIF raw, …); out, `.npy`, `.mat`, GIFTI and NIfTI (a map on the same
  geometry as an input). Arrays are `[elements x samples]` (channels or vertices by samples) in every format; a recording's units are SI.
- **Metadata** a node needs that the array does not carry: `--sfreq` for a streamed input read from `.npy` or `.mat` (a recording file carries its own), and
  `--unit`. Value type: real or complex from the data type; a vector3 or ℍ field is declared with `--value PORT=vector3`.
- **Recordings are streamed.** An input from a recording format is read chunk by chunk through io's chunk service over the format's reader, with the halo the
  checked graph declares; **the time range is required** (`--from/--to` or `--all`), as for every computation over a recording. A streamed output is written
  as it comes (the `.npy` header is sized from the checked output type). An `.npy`, `.mat`, GIFTI or NIfTI input is an array handed whole (`apply`).
- **Provenance.** Every output gets a sidecar `<output>.json`: nxr-neuro's version, the node or graph (name, version, sha256 of its text; for a composite the
  expanded graph), every parameter (defaults included), every input (path, selector, sha256, the type it was read as), the span read, each node's `record`,
  and the time taken. GIFTI and NIfTI outputs also carry it in their own metadata (`nxr-neuro:producer`), as the exporters do. No cache: `compute` stores
  nothing in a dataset (question 10). An existing output is refused unless `--replace`.

Examples:

```
nxr-neuro compute filter.bandpass rest.edf rest_alpha.npy --lower 8 --upper 12 --from 0 --to 60
nxr-neuro compute welch.psd x.npy psd.npy --sfreq 600 --segment-samples 512
nxr-neuro compute welch.psd run.fif#MEG psd.mat --all --frequencies freqs.npy
nxr-neuro compute manifold.eigenmodes lh.pial.surf.gii modes.npy --k 200 --eigenvalues evals.npy
nxr-neuro compute phase.current_of_band x.npy K.npy cortex.surf.gii j.npy p.npy --sfreq 600 --band '[8,12]'
nxr-neuro compute my_chain.yaml x.npy y.npy --sfreq 1000 --band '[15,29]'
nxr-neuro compute welch.psd --help
```

### 6.2 `run <pipeline>`: io around the executor

```
nxr-neuro run <pipeline | definition.yaml> --dataset D --subject S [--<parameter> VALUE]... (--from S --to S | --all)
              [--replace] [--dry-run] [--json]
nxr-neuro run <pipeline> --help
nxr-neuro list pipelines
```

- **One verb for every pipeline.** A name is looked up in one table: the hand-wired C++ pipelines (today's `compute <what>` commands, each dispatched to its
  `<x>_compute` function with its parser unchanged) and the shipped YAML definitions; a path (`*.yaml`, or containing `/`) is the user's definition. A name in
  both tables fails a build test.
- **Options.** A hand-wired pipeline's options are its own (unchanged). A YAML definition's parameters become `--<parameter> VALUE` (VALUE read as YAML);
  `--set NAME=VALUE` goes. A parameter named as a reserved option (`dataset`, `subject`, `from`, `to`, `all`, `replace`, `dry-run`, `json`, `help`) is refused
  when the definition is read.
- **Help** (`run <pipeline> --help`) is generated: for a hand-wired pipeline from the command table (as today), for a definition from its parameters, which
  declare a default, a meaning and a unit each (the long form below). Shipped definitions must declare all three (the test `cli_helptext` extends to them); a
  user's definition may use the short form `name: default`, and help then says the parameter is undocumented.
- **The time range** is `--from/--to` or `--all` for every pipeline over a recording, YAML ones included (`--chunks`, `--start`, `--stop` become hidden
  developer options, as for the hand-wired ones).

The parameters' long form in a definition (and in a composite):

```yaml
parameters:
  band:        {default: [8.0, 16.0], meaning: "the time level's band (its resolves_hz)", unit: Hz}
  space_level: {default: 5, meaning: "the surface ladder's level: its Weyl band is the spatial band, its tiles the spatial partition"}
  enter_sd:    {default: 1.5, meaning: "the threshold a burst enters at, above the baseline's median", unit: "robust SD"}
```

Examples:

```
nxr-neuro run psd --dataset store/demo --subject S1 --recording rest --from 0:10 --to 1:10
nxr-neuro run eigenmodes --dataset store/demo --subject S1 --surface pial --modes 200
nxr-neuro run global_bursts --dataset store/demo --subject S1 --all --band '[15,29]' --space-level 5
nxr-neuro run ./my_pipeline.yaml --dataset store/demo --subject S1 --all --dry-run
nxr-neuro run global_bursts --help
```

### 6.3 Old spellings: a clean switch, and what callers change

nxr-neuro is not released, so the hidden aliases of `2026-11-16-command-line.md` (section 3) are removed, not kept: an old spelling is refused, and the error
names the new one (made from the same mapping table; no alias runs). The mapping:

| old | new |
|---|---|
| `compute <what>` (every computation of the command-line spec's table: `psd`, `sourcepsd`, `eigenmodes`, `measure`, `null`, …) | `run <what>`, the same options |
| the old two-word spellings (`psd compute`, `eigen solve`, `measure compute`, …) | `run <what>` |
| `run DEFINITION --set NAME=VALUE --chunks …` | `run <pipeline> --NAME VALUE --from … --to …` / `--all` |
| `run --list` | `list pipelines` (and `list nodes`) |
| `nn.ops.<group>.<function>` (Python), `nxr.ops.<group>.<function>` (MATLAB) | `nn.compute.<group>.<node>`, `nxr.compute.<group>.<node>` |
| `nn.compute.<verb>(sub, …)` (planned, never built) | `nn.run('<pipeline>', …)` |

Cache keys name the pipeline (`psd`), not the spelling: results stored under the old spelling stay cache hits.

Callers outside the repository to change:
- **The Studio's analysis route** (`/api/compute`): its allowlist (`COMPUTE_COMMANDS`, in the Studio's backend service) holds old spellings
  (`eigen solve`, `frames compute`, `psd compute`, …) that the WebAssembly module's command line runs. Each becomes `run <what>`; the route keeps its
  name. Changed with the Studio's next update to nxr-neuro, together with its test that calls `eigen solve` through the route.

Inside the repository (each in the PR that switches it): the release smoke test (`.github/smoke/smoke.sh`: `compute eigenmodes` → `run eigenmodes`,
`compute --help` → `run --help`), `.github/release/README.txt`, `README.md`, `docs/reference.md`, the comments at the top of each shipped definition, the
CLI tests, the MATLAB and WebAssembly smoke and parity scripts that call the command line.

## 7. The bindings: the same verbs in Python and MATLAB

Python is `import nxr_neuro as nn`; MATLAB's package is `nxr`. Both have exactly the command line's verbs, generated from the same registry and pipeline
table, with the same names and defaults; docstrings and MATLAB help are the `ParamSpec`/`PortSpec` text (meaning, default, unit).

| verb | Python | MATLAB |
|---|---|---|
| one node | `nn.compute.welch.psd(x, sfreq=600.0, segment_samples=512)` | `r = nxr.compute.welch.psd(x, 'sfreq', 600, 'segment_samples', 512)` |
| any node or graph by name, path or object | `nn.compute("welch.psd", x, sfreq=600.0)`, `nn.compute(g, x=x, kernel=K, band=[8, 12], sfreq=600.0)` | `r = nxr.compute('welch.psd', x, 'sfreq', 600)`, `r = nxr.compute(g, 'x', x, 'kernel', K, 'band', [8 12])` |
| a pipeline | `nn.run("psd", ds, "sub-01", recording="rest", all=True)` | `r = nxr.run('psd', ds, 'sub-01', 'recording', 'rest', 'all', true)` |

```python
import nxr_neuro as nn

f = nn.compute.filter.bandpass(x, sfreq=1000.0, lower=8.0, upper=12.0)    # arrays in, a Result out: f.y (the output port), f.record (provenance)
psd = nn.compute.welch.psd(x, sfreq=1000.0, good=mask)                     # psd.frequencies, psd.psd, psd.segments, psd.segments_skipped

g = nn.Graph.load("my_chain.yaml")                                         # or nn.Graph({...}) from a dict, or a composite's name
r = nn.compute(g, x=x, kernel=K, surface=(V, F), sfreq=600.0, band=[8, 12])
r.current, r.power                                                         # the output ports, NumPy, the input's precision

members = tiles_of_time                                                    # a partition handed as an array: member index per sample, -1 none
p = nn.compute.field.power(z)                                             # |z|² per element per sample
m = nn.compute("measure.over_partition", p.y, partition=members)          # per member: the weighted sums and their weight (the ingredients)

with nn.compute.stream(g, sfreq=600.0, band=[8, 12]) as s:                 # blocks handed one at a time (question 11): the same numbers as one call
    for x_k in blocks:
        out_k = s.block(x_k)
    final = s.finish()

nn.list_nodes("welch")                                                     # the registry: names, ports, parameters
r = nn.run("global_bursts", nn.open("study/"), "sub-01", start=0.0, stop=300.0, band=[15, 29])
r.cache, r.read()                                                          # "hit" | "computed"; the stored values with their axes and provenance
```

```matlab
f   = nxr.compute.filter.bandpass(x, 'sfreq', 1000, 'lower', 8, 'upper', 12);   % f.y, f.record
psd = nxr.compute.welch.psd(x, 'sfreq', 1000);
g   = nxr.Graph.load('my_chain.yaml');
r   = nxr.compute(g, 'x', x, 'kernel', K, 'sfreq', 600, 'band', [8 12]);
r   = nxr.run('global_bursts', nxr.open('study/'), 'sub-01', 'from', 0, 'to', 300, 'band', [15 29]);
```

- **Generated.** `nn.compute.<group>.<node>` is made at import from the registry the module carries (signature, defaults, docstring); MATLAB's
  `+nxr/+compute/+<group>/<node>.m` files are generated at build from the same table. A test checks that every registered node and composite has its
  function in both faces, and nothing more.
- **Arrays** keep the first slice's conventions: `[elements x samples]`, float32 or float64, the result in the input's precision; MATLAB's indices
  1-based. Metadata keywords are the command line's: `sfreq`, `unit`, `value`. A partition is an integer array.
- **The merged binding is renamed:** Python `nn.ops` → `nn.compute`; MATLAB `nxr.ops` (open as the MATLAB first slice) → `nxr.compute`. The planned
  `nn.compute.<verb>(sub, …)` and `nxrneuro.compute('psd', sub, …)` of the interoperability spec become `nn.run(...)` and `nxr.run(...)`.
- **`run` in the bindings** calls the same library function as `run` on the command line (`pipelines`, through io's task API), so a result made from
  any face is the same cached result.

## 8. What moves where: the PR plan, sequenced with the io record API migration

The record API migration (`2026-11-17-io-record-api.md`, section 0) is in progress: steps 1 (the repositories) and 2 (the task API) have merged; steps 3
(the small steps), 4 (the runner: `runner.h`/`registry.h` take the `Subject`), 5 to 7 (partitions and pyramid, `meeg.cpp`, the command line), 8 (the
WebAssembly binding), 9 (tests) and 10 (close) follow. That migration changes **how** the runner and the steps find and store; this plan changes
**where** the step logic lives. To avoid two branches rewriting the same lines, every PR here that touches `runner.cpp`, `registry.cpp` or a `*_steps.cpp`
waits for the migration step that touches it, and builds on the task API it leaves.

| # | PR | contents | waits for | runs beside |
|---|---|---|---|---|
| C0 | this spec | the model, the interfaces, the grammar, the plan | — | everything |
| C1 | `nxr_graph`: the executor, standalone | `pipelines/graph/` with `PortType`, `Block`/`Array` (moved; `registry.h` re-exports them), `NodeTraits` (moved), `NodeSpec`, `Graph`, `check`, `Execution`, `apply`, `dispatch` (moved; the guard follows it); the layering rule (no io include); tests with synthetic toy nodes; the dependency table's new row. **New files only**; the runner is untouched | — | migration 3 to 10 |
| C2 | the bindings' rename | Python `nn.ops` → `nn.compute` (hand-written functions kept for now), MATLAB `nxr.ops` → `nxr.compute` (in the open MATLAB first slice, before it merges); docs | — | everything (the migration touches no Python or MATLAB code) |
| C3 | `run` for every pipeline; the clean switch | the command table: every `compute <what>` moves to `run <what>`; hidden aliases removed, old spellings refused with the new name; `run` takes `--from/--to/--all` and `--<parameter>`; the definitions' long-form parameters (`definition.h` parse) and the 15 shipped definitions documented; `list pipelines`; generated `run <pipeline> --help`; the name-collision test; smoke test and docs | — | migration 3 to 6. It touches `cli/`; migration step 7 touches only lookup lines in `cli/` (`run_command.cpp`'s `--dry-run` subject lookup). Whichever lands second rebases; C3 should land first if both are ready |
| C4 | nodes without io | every step's `plan` and `prepare` stop reading the database or the subject: lookups become declared inputs bound by the runner (the glued surface, the ladder's tiling, the stored eigenmodes), `resolve` becomes the runner's binding step; the spatial band's "compute and store the eigenmodes when absent" leaves the node (question 6). One PR per file: `registry.cpp` (37 such reads), `spatial_events_steps.cpp` (14), `crossings_steps.cpp` (7), `extent_steps.cpp` (6), `cores_steps.cpp` (5), `surrogate_steps.cpp` (4); the shipped definitions name the new inputs | migration 3 (small steps) and 4 (runner): they rewrite these same lookups onto `task::find`; C4 then moves those calls from the node into the runner | migration 5 to 10 |
| C5 | the runner split | the step logic of `runner.cpp`'s `execute` (halo composition, the sub-block and frame-block tails, start/step/finish, before/after) moves into `Execution`; the steps' code moves to `pipelines/graph/nodes/` as `NodeSpec`s; the runner keeps find → bind (task `Node` → `PortType`, constants read by name) → `haloed_loop` calling `Execution::block` → store; `StepSpec`, `PlanContext`, `ExecContext` go | C4; migration 4 | migration 5 to 10 (they touch other files) |
| C6 | the registry's names and the pure functions as nodes | grouped names for every node (table in 8.1); the bound pure functions (`welch.psd`, `welch.cross_spectrum`, `filter.*`, `pca.*`, `manifold.eigenmodes`, `mesh.gradient`) registered as nodes with their goldens; the shipped definitions use the new names; nxr's reference definitions follow (question 2) | C5 | — |
| C7 | composite nodes | `wrap`, expansion in `check`, the composite YAML, `pipelines/graph/nodes/*.yaml`, the search path, `list nodes`, provenance of composites | C6 | C8 |
| C8 | `compute` on files | the `compute` verb: ports positional, parameters from the registry, the format readers and writers, streamed recording inputs, the sidecar; man pages and completion from the registry | C6 (C7 for composites) | C7 |
| C9 | the bindings generated | `nn.compute.<group>.<node>`, `nn.compute(...)`, `nn.compute.stream`, `nn.Graph`, `nn.list_nodes`; `nn.run`; the MATLAB twins and generated `+compute` files; the coverage test | C6, C3; `run` needs the task API (merged) | — |
| C10 | close | `ARCHITECTURE.md` (the executor at the pure level, the runner's four jobs), `CONTRIBUTING.md` (adding a node; adding a verb is adding a pipeline), `docs/reference.md` (the `compute` and `run` sections, the node list) | C9 | — |

C1, C2 and C3 can start at once and in parallel with the migration. C4 and C5 are the only PRs that rewrite the migration's files, and each waits for the
migration step that last touches them. Migration steps 5 (partitions and pyramid) and 6 (`meeg.cpp`) touch the hand-wired pipelines, which this spec does
not convert, so they never meet C4 or C5. Main stays green at each step: C1 adds; C3 keeps every pipeline's JSON output identical under its new spelling
(tested against the old spelling's output, captured before the switch); C4 and C5 keep the runner's tests and the definitions' outputs bit-for-bit.

### 8.1 Proposed grouped names for today's steps

| today | node | | today | node |
|---|---|---|---|---|
| `band_filter` | `band.filter` | | `measure` | `measure.over_partition` |
| `analytic` | `band.analytic` | | `collect` | `measure.collect` |
| `envelope` | `band.envelope` | | `summarize` | `statistic.summary` |
| `frames` | `tiling.frames` | | `reduce_cells` | `cells.reduce` |
| `apply_kernel` | `kernel.apply` | | `cell_extent`, `extent_measure` | `cells.extent`, `cells.extent_measure` |
| `form` | `field.form` | | `risefall` | `bursts.risefall` |
| `magnitude` | `field.magnitude` | | `fit_gate`, `burst_stages` | `bursts.fit_gate`, `bursts.stages` |
| `power` | `field.power` | | `detect_spans` | `spans.detect` |
| `phase_current` | `phase.current` | | `measure_events` | `events.measure` |
| `gradient_power` | `phase.gradient_power` | | `core_histogram`, `cores`, `tracks` | `cores.histogram`, `cores.detect`, `cores.tracks` |
| `tile_max` | `tiles.max` | | `tile_crossings`, `power_series` | `tiles.crossings`, `tiles.power_series` |
| `lagfree_surrogate` | `surrogate.lagfree` | | (the bound pure functions) | `welch.psd`, `welch.cross_spectrum`, `filter.lowpass` … `filter.notch`, `pca.pca`, `pca.reconstruct`, `manifold.eigenmodes` |

## 9. Tests (scoped, through `tools/tests.py`; synthetic data only)

**The executor (C1, C5).**
- `check` refuses each bad arrow with the use, the port and the reason: a value type, rows against an operator, a manifold, a unit, `absq` as a phase input,
  a linear map after a non-linear node (D106), a chunk node in `after`, a block into a `before` node, an unknown node (the near names given), a node with
  undeclared traits.
- The halo composed over a chain of windowed nodes equals the sum rule of 3.3; a toy chain run on blocks with that halo equals the whole array on every column.
- **Block independence:** for every registered chunk and accumulate node, `apply` on a whole array equals `Execution::block` over blocks of several
  widths (including width 1 for pointwise nodes), to the tolerance of the folds' rounding; `block_samples` and `block_frames` likewise.
- **State:** a stateful toy node (a running sum) and an accumulate one give the same result as the whole array; blocks out of time order are refused
  for them in a block-parallel context.
- `dispatch_is_the_only_call_path` follows the moved files; the layering rule: `pipelines/graph/` includes no io header.

**Composites (C7).** A composite equals its flat graph bit-for-bit; its derived kind, traits and halo; a cycle refused; a name that collides with a node or
a pipeline refused; a port named as a parameter refused; the provenance names the composite, its version and sha256, and the expanded graph.

**`run` (C3, C5).** Every hand-wired pipeline runs under `run <what>` with the same JSON lines as its old spelling; every old spelling is refused naming the
new one; the shipped definitions run with `--from/--to/--all` and `--<parameter>`; `run <pipeline> --help` prints every parameter's meaning, default and
unit (`cli_helptext` covers definitions); no name is in both the hand-wired table and the shipped definitions; a reserved parameter name is refused; the
runner's existing plan, execute and parity tests pass unchanged after C5 (the definitions' outputs bit-for-bit).

**`compute` (C8).** Each format in and out on synthetic files (round trip; precision follows the input); the sidecar's fields; a recording input streamed
equals the same samples handed as an `.npy`; a missing time range for a recording refused; an existing output refused without `--replace`;
`compute <node> --help` lists every port and parameter. **Parity across verbs:** `compute welch.psd` on the samples of a span equals what `run psd`
stores for that span (both call the same compute function).

**Bindings (C9).** Every registry node and composite has `nn.compute.<group>.<node>` and its MATLAB file, with the registry's signature; a node called from
Python, from MATLAB and from `compute` on the same `.npy` gives the same numbers; `nn.compute.stream` equals one call; `nn.run` and `run` give the same
cache key (the second is a hit).

## 10. Open questions, each with a recommendation

| # | question | options | recommendation |
|---|---|---|---|
| 1 | Where the executor lives | `pipelines/graph/` as its own target `nxr_graph` (compute, meeg; no io) · a new top-level folder | **`pipelines/graph/`, own target**: one new row in the dependency table and a layering rule; no new top-level folder |
| 2 | Grouped names and nxr's shared step names (chain-graph spec, decision 3: the same names in both tools) | rename in both tools together · nxr-neuro only, with nxr's name kept per node as an alias | **nxr-neuro only** (decided 2026-11-17): nxr is being phased out in favour of the bindings, so it is not changed; the parity tests map nxr's step names to the grouped names. The grouped names of 8.1 go to the maintainers for confirmation when the renaming PR (C6) is opened |
| 3 | Composites: expanded at check, or run as nested executions | expand (TSL inlines) · nest | **expand**: one halo, one dispatch point, fusable later; the provenance keeps the composite |
| 4 | A composite with both streamed and array outputs | allow · refuse | **allow**: after expansion it is an ordinary graph; its kind is `accumulate` and its streamed outputs are handed back per block |
| 5 | Where a user's composites are found | a search path (`--nodes DIR`, `NXR_NEURO_NODES`) · paths only | **both**: a path always works; the search path for reuse; a user node never shadows a built-in one; the file's path and sha256 in the provenance |
| 6 | A node that today computes and stores a missing input (the spatial band's eigenmodes) | the runner runs the `eigenmodes` pipeline first when a definition marks the input `make: eigenmodes` · the user runs it first, the run refused with the command to run · the node falls back to its series route (announced) | **the runner makes it, with a message (decided earlier, 2026-11-13: eigenmode first; when the stored eigenmodes are absent they are computed, with a message; the Chebyshev series is the announced fallback, never silent)**: a definition marks the input `make: eigenmodes`; the runner runs that pipeline first as its own run (its own provenance and cache key, so nothing is stored hidden inside the outer run), prints which input it made and why, then continues. `transform: chebyshev` stays a named option |
| 7 | A definition's parameters on the command line | `--<parameter> VALUE` · `--set NAME=VALUE` | **`--<parameter> VALUE`**, as the hand-wired pipelines' options, so help and completion treat both alike |
| 8 | Old spellings | refused with the new name · refused as unknown | **refused with the new name** (no alias runs), until the first release |
| 9 | A node passed as a parameter (TSL's `seedNode(value)`) | not now · a `node:` parameter type | **not now**: no use needs it yet; a composite covers reuse |
| 10 | Caching in `compute` | none · a cache beside the output | **none**: `compute` stores nothing; the sidecar holds what a cache key would, so a later `import` of a result keeps its provenance |
| 11 | Handing blocks one at a time from the bindings (`nn.compute.stream`) | yes · whole arrays only | **yes, in C9**: it is the executor's `block`/`finish`, already tested for block independence; it lets a binding stream a long array it reads itself |
| 12 | MATLAB: `nxr.compute(...)` as a function beside the `+nxr/+compute` package | both · the string form only | **both, if MATLAB resolves them** (checked in C9's CI); otherwise `nxr.compute('<node>', …)` is the one form and the generated files go under `+nxr/+node` |
| 13 | `setup` and `after` merged into one `array` kind | merge · keep both | **merge**: the section a definition puts a node in already says when it runs; `check` refuses a `before` node that reads the loop |

## Checklist (the architecture policy)

- **Can each new compute function be called with a matrix from anywhere?** Yes, more than before: every node, and every graph of nodes, is callable on arrays
  (`compute`, the bindings) and its plan reads types, never rows.
- **Can the pipeline run the same chain on another input of the same kind just by naming it?** Yes: `run` finds the inputs the definition names; `compute`
  takes any array or file of the right type.
- **Is every loop over samples or chunks, and every file access, inside io?** Yes: the chunk loop is io's (`haloed_loop` in `run`, the chunk service over a
  format reader in `compute`); the executor is handed blocks; files are read and written by `io/formats`.
- **Is every choice of which data (signal, source, span) made by the caller?** Yes: `run` takes the span and the inputs from the caller; `compute` takes the
  files and the span; no node picks or stores its input (C4 removes the last ones that did).
