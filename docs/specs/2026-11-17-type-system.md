# The pipeline type system: the data model's classes as port types, checked before any data flows (spec, 2026-11-17)

Status: **design, for the maintainers' review before code.** It completes `ARCHITECTURE.md` part 2 ("type-check everything before any data flows")
and `2026-11-17-compute-and-run.md` §3.1 and §3.3 (`PortType`, `check`). It builds on `2026-11-17-definition-outputs.md` (the six output kinds),
`docs/audits/2026-11-17-format-mapping-and-coordinates.md` (covariance a statistic, the whitener an operator, transforms as transition operators, the
source space a selection, sensor types as partitions; its closing section "For the coming type system") and `2026-11-17-geometry-central-switch.md`
§5.2 (G5: `PortType::spin`, `PortType::gauge`). It is designed from nxr-neuro's own pipelines: the inventory in appendix A is its source.

## 1. Why

What a port carries is known today by three means, none of them shared:

- **`PortType`** (`pipelines/graph/include/nxr/graph/types.h`): class, value type, rows, axis, rate, unit, and `on` / `to`, a manifold **label**
  ("sensors", "cortex"). A label says what kind of thing a manifold is, not which one: two subjects' cortices, or a cortex and its glued twin, have
  the same label.
- **`facts`**, a free JSON object each step's `plan` writes for the steps after it: `made_by`, `nonlinear`, `band`, `spatial_band.level`, `over`,
  `axis`, `levels`, `glued`, `of_complex`. A later step reads these by key. The meaning of a key lives in the two steps that agree on it.
- **Per-step checks.** The registered steps make **282** checks in 39 steps (appendix A.2, counted by `tools/type_inventory.py`). About **158** of
  them are type checks that several steps repeat in their own words: the class or producer of an input (35), the manifold it lies on (37), the
  partition, ladder or level it is over (54), the band or rate (13), the value type (13), linearity (D106, 2), a stored record (4). The other 124
  check parameters and array shapes; they stay in the nodes. The ~35 hand-wired verbs (appendix A.3) make their own checks again, mostly the same
  ones: a kernel on other channels than the recording, a kernel's rows against its surface's vertices, a fixed kernel where a free one is needed, a
  tile level or a band that does not match, a selection on another time line.

The type system replaces the label, the cross-node facts and the repeated type checks with **one** description of what flows (the data model's
classes as types, with the manifold itself in the type) and **one** checker that every graph goes through, in `run` and in `compute` alike, before
any array is read. It is kept as small as catches the mistakes nxr-neuro's pipelines can make: there is no units algebra, and no shape arithmetic
beyond what the manifolds already say.

## 2. What flows: the inventory's findings

Appendix A lists, for every shipped definition and every hand-wired verb, what flows through each port. Read as a whole, it says:

| what flows | where | class |
|---|---|---|
| a recording: values on sensors × time, streamed | every recording pipeline | field |
| a linear map between manifolds: inverse kernel, forward, eigenvectors, a spatial band's filter, an interpolation onto frames | `apply_kernel`, `frames`, `inverse`, `hodgepower`, `pointspread` | operator |
| a change of coordinate system on one space: device → head, voxel → scs | importers, `headmodel` | operator, kind `transition` |
| a grid on a manifold (tiles of a level, a constant-Q level, a frame tiling, Welch segments, analytic windows, a joint or product grid of surface × time, channel types) or an irregular selection (detected events, tracks) | `measure`, `measure_events`, `psd`, `cellquadrature`, `pyramid` | selection (a grid, or an irregular one) |
| a family of partitions indexed by level, each level with its resolution (a time band, a spatial band) | tile ladder, constant-Q ladder | selection, type `ladder` |
| a subset of a manifold's elements: a kernel's channels, a source space, a bad span | kernels' `from_selection`, `exclude` | selection |
| one value per member of a partition | `measure`, every detection's per-event values | measurement |
| what is computed from a partition's members: a PSD, a cross-spectrum, a covariance, a histogram, a baseline, summaries | `summarize`, `psd`, `baseline`, `noisecov` | statistic |
| a line or a space made by the computation: a frequency line, a mode (eigen) axis, a frame line, a glued surface, an index line | `psd`, `eigenmodes`, `frames`, `apply_kernel glued`, `speedspectrum` | manifold |
| the stack of a field's spatial levels (`levels`), a free current's three components | `apply_kernel levels`, `at_level`, `scale` | the value's components |

**Are the six classes enough?** As what is **stored**, yes: nothing in the inventory needs a seventh class. As what **flows**, three things the
classes contain must be first-class in the **type**, or the checks cannot be shared:

1. **The manifold's identity, with products, restrictions and member sets.** A field is on a product of factors (sensors × time, cortex × frames);
   a kernel maps from a *restriction* of a manifold (the sensors of its `from_selection`) to another manifold; and every measurement lies on the
   **member set of a partition**, written `members(P)`: the time tiles of a level, the cells of a joint partition, the events of a detection. The
   member set is a manifold in every sense the checker needs. A selection can lie on it: `cell_extent` labels the cells of a joint partition, and
   `tracks` labels the cores. It is not a stored manifold of its own: its storage is the partition (dictionary and count) and the measurement rows
   keyed by member. The **frame line** of `frames` is exactly `members(frame tiling)`, so the retiming needs no special case.
2. **The partition property of a selection.** `measure` is the only reduction, and it takes **any** selection on factors of the field: a region, a
   time range, a channel set, a tiling. Measuring one region or one time range is a selection of one member. Disjoint members (a **partition**: `set`
   with a dictionary, `spans` that do not overlap, a ladder's level, a product of two partitions) are required only where members are compared or
   grouped: `within` (grouping by another selection) and a roll-up through a grid's hierarchy (`summarize`). Two kinds of selection exist. A **grid**
   partitions the manifold and is defined before any data: chunk grids, constant-Q ladders, dyadic spatial tiles, frame tilings, and products of grids; its levels
   are a closed-form hierarchy, so only a grid rolls up. An **irregular** selection partitions the field (thresholding a measured signal): bursts, events, scouts,
   atlases; its members are sparse, may overlap, have any extent and no levels, and are measured member by member; statistics across them are `summary`
   statistics of their measurements, never a roll-up. The type carries the kind (`SelectionShape::partition`, true for a grid; a property and not a type: the five selection types are point, span, distance, indices and set), and a ladder's level carries its
   **resolution** (the time band of a constant-Q level, the spatial band of a tile level: "the level IS the band"), which is what most of today's
   level checks compare.
3. **The transition**, an operator between two coordinate systems of one space rather than between two manifolds. Applied to positions, it changes
   the manifold's coordinate system and nothing else. The audit already asks for it.
4. **An operator a node defines, and the line it makes.** A change of basis (the FFT, a wavelet, an eigenmode projection) is an `apply` of an operator
   the node itself defines (from the time line to a frequency line, from the surface to the mode axis), not a free-standing transform. The value it
   makes lies on the line that operator makes: a spectrum is on a frequency line (bin k at k × rate / n Hz, origin 0, unit Hz), a projection on the
   mode axis. Such a line is a manifold the node makes (the inventory's "a line or a space made by the computation"), with its own identity, so a
   later node reads a band, a bin or a mode on it by type, and a reduction over a block axis that is only a number is refused. The made line is a **factor**
   of the output's domain: the output of `fft.spectrum` lies on the product `channels × frequency`, the Studio's product manifold (D37: manifold type
   `product`, geometry form `product`, a `geometry_dimension` per factor: 0 for the channels, 1 for the line; cells per factor, `0,1`). There is no `Axis::block`:
   a value a chunk node makes once per block is a value whose last factor is a line the node made (`PortType::per_block()`, derived from the domain). The
   line's spacing is `rate / n`, with `n` the samples of the block it was made from; `n` rides on the block value (`Block::line_samples`) because the bins do
   not tell it (an odd and an even `n` make the same count of bins).

Two attributes, today buried in `facts`, become fields of a field's type because most of the repeated checks compare them: the **quantity** (what
the values are: `recording`, `envelope`, `amplitude`, `power`, `phase_current`, `relative power`, `curl_tan`; the stored field's `kind`, a
measurement's `measure`), which replaces every `made_by` check, and the **resolution** (the time band and the spatial level), which replaces the
`band` / `spatial_band` / `levels` comparisons.

What is **not** made first-class: recipes (a dictionary entry, provenance), cross-subject contributions (out of scope here: `reduce` stays a verb
until its own spec), the byte-less fields of `phasegradient`, `opticalflow` and `pyramid` (they become measurements, which they are), and
statistic axes made of named entries (bands, patterns, tests). A named-entry axis is a small manifold the node makes, of kind `points`, with no
geometry (§3.1).

## 3. The types

### 3.1 Manifolds as terms

```cpp
namespace nxr::graph {

enum class ManifoldKind { points, line, surface, volume, curve, graph, members };
/** what a line or member set is FOR, as far as the checker cares: the factor a stream walks, and what a measurement's member set came from */
enum class Role { none, time, frames, time_tiles, frequency, modes, levels };

/** a chart on a space, with a unit (the schema's coordinate_system): {"head", "scs", "m"} · {"time", "clock", "s"} · {"frequency", "hz", "Hz"} ·
 *  {"spectrum:laplaceBeltrami", "eigenvalue", "1/m^2"}. Two manifolds are comparable in position only when their systems are of one space. */
struct CoordinateSystem { std::string space, name, unit; };

/** A MANIFOLD AS A TERM: which manifold, known or not yet known. Equality is structural: two ids are equal when the ids are; a member set when its
 *  partitions are; a restriction when the manifold and the selection are. */
struct Manifold {
  enum class Form {
    id,          // a stored manifold, by id (run: resolved when the inputs are found)
    anonymous,   // an array's manifold in `compute`: equal only to itself (or to another the caller declares the same)
    variable,    // a node signature's M, T: bound by unification, never left in a checked graph
    made,        // made by a node in this graph (a frequency line, a mode axis, a frame line written as a line): keyed "<use>.<output>"; a factor of the output's domain
    restricted,  // `of` restricted to the elements of the selection `by` (a kernel's channels, a source space)
    members,     // the member set of the partition `by`
    derived      // a stored manifold related to `of` by `relation` (`glued`: the glued surface of a cortex), resolved in run, structural in compute
  };
  Form form;
  std::string key;                         // the id · the anonymous number · the variable's name · "<use>.<output>" · the relation's name
  ManifoldKind kind = ManifoldKind::points;
  Role role = Role::none;
  std::optional<CoordinateSystem> cs;      // the system its positions are in; none for a manifold without geometry (members, an index line)
  std::int64_t n = 0;                      // its vertices (0: not known before data, as a stream's time line)
  std::optional<double> spacing;           // a closed-form line's spacing (1 / the sample rate on a time line; the frame rate on a frame line)
  std::optional<double> origin;            // made line: the coordinate of vertex 0 (0 Hz for a one-sided spectrum)
  std::optional<double> rate;              // made line: the rate of the line it was made from; its spacing is rate / n, n the block's samples (carried on the block)
  std::string unit;                        // made line: the unit of its coordinate ("Hz")
  std::shared_ptr<const Manifold> of;      // restricted, derived: the manifold it is of
  std::string by;                          // restricted, members: the selection's identity (a stored id, or "<use>.<output>")
};
using ManifoldRef = std::shared_ptr<const Manifold>;

/** where values lie: a PRODUCT MANIFOLD's factors in axis order (one factor when it is not a product), and the cells (the schema's key, one number per factor: "0", "1", "2", "1,0").
 *  A node's operator makes a factor: a spectrum lies on channels × the frequency line the FFT made. As the Studio's products (D37), the first factor varies fastest in a product's
 *  code (D100 `productCode`). */
struct Domain {
  std::vector<ManifoldRef> factors;
  std::string cells = "0";
  /** the factor a stream walks: the last, when its role is time or frames; none for an array */
  const Manifold* streamed() const;
};
}
```

### 3.2 `PortType`, extended

`PortType` keeps every member it has (class, value type, rows, axis, rate, unit, `nonlinear`, `facts`) so that C1's executor and today's plans
keep compiling; `on` and `to` change from labels to domains; G5's `spin` and `gauge` are part of it; four members are added. `rows`, `axis` and
`sfreq` become derived (from `on`) and are kept as caches.

```cpp
enum class PortClass { field, operator_, selection, measurement, statistic, manifold, constant };   // the six classes; `partition` leaves (it is a selection's property)
enum class ValueType { real, complex, vector3, quaternion, hermitian3, monogenic, integer };

/** what a field is band-limited to, or what a ladder's level resolves (the level IS the band) */
struct Resolution {
  std::optional<std::array<double, 2>> band;       // Hz, with its shape ("morlet", "meyer")
  std::string band_shape;
  std::string ladder;                               // the spatial ladder's identity, and its level(s): one level, or a stack (the value's components)
  std::vector<int> levels;
  std::string spatial_shape;                        // "mexhat" · "meyer"
};

/** a selection's shape: what the members are */
struct SelectionShape {
  std::string type;          // the schema's: point · span · spans · indices · kernel · set · ladder · tiling (closed-form spans)
  bool partition = false;    // a GRID (a property, not a type): defined before data, hierarchy in closed form; only a grid rolls up. false: an irregular selection (detected events, scouts, atlases), measured member by member
  std::string identity;      // a stored id, or "<use>.<output>" for one this graph makes
  std::string detected_over; // detected events: the identity of the partition whose members they were found on
};

/** what a statistic is OF: the quantity, the resolution and the domain of the field or measurement before the reduction */
struct OfType { std::string quantity; Resolution res; Domain on; };

struct PortType {
  PortClass cls = PortClass::field;
  ValueType value = ValueType::real;
  int components = 1;              // a vertex's components (3 for vector3) times a stack of levels
  Domain on;                       // field, measurement: where the values lie · selection: what it indexes · statistic: its axes · manifold: itself · operator: its source
  Domain to;                       // operator: its target
  std::string quantity;            // field: its kind (what the values are) · measurement: the measure · statistic: the measure · operator: its kind · selection: its type name
  Resolution res;                  // field, measurement: band and spatial level · selection: a ladder level's or a joint partition's
  bool nonlinear = false;          // D106, kept (true: a linear map may not follow)
  int spin = 0;                    // G5: 0 scalar, 1 tangent vector, n an n-RoSy, -1 a conjugate
  std::string gauge;               // G5: "" for spin 0 and for vector3; else the gauge id
  std::string unit;                // SI, carried; compared for equality only where a node asks
  std::optional<SelectionShape> sel;
  std::optional<OfType> of;                                          // statistic
  std::optional<std::pair<CoordinateSystem, CoordinateSystem>> transition;  // operator kind transition: from → to, one space
  // kept from C1 (derived from `on`, cached): rows, axis, sfreq
  std::int64_t rows = 0; Axis axis = Axis::none; std::optional<double> sfreq;
  Json facts = Json::object();     // a node's own facts for ITS prepare (a ladder's tiling, a channel list); never read by another node's check
};
```

**`facts` is narrowed, not removed.** A node may still record what its own `prepare` needs (a tiling's origin, a channel list), but no node's check
may read a key another node wrote. Every fact a check compared today becomes a member above: `made_by` → `quantity`; `nonlinear` → `nonlinear`;
`band`, `spatial_band`, `levels` → `res`; `over` → the measurement's domain (`members(P)`) and `sel->detected_over`; `axis: frames` → the frame
line's role; `glued` → the derived manifold; `of_complex` → `quantity` (`power` of a complex field is `power`; of a real field, `square`). A test
(§11) greps the nodes for a `facts` read of a key they do not write.

### 3.3 The printed form

One printer and one parser, used by `--dry-run`, `--help`, error messages, composites' declared ports (`2026-11-17-compute-and-run.md` §4) and the
sidecar of `compute` on files:

```
field<cortex(kernel) × frames(tiling); complex vector3; current; band 8-12 Hz morlet; level tiles@5>
measurement<members(cells_l5); amplitude; band 8-16 Hz; level tiles@5>
selection<spans on time(rest); partition; detected over ladder@8-16>
statistic<summaries [median, mad]; axes groups(cells_l5); of amplitude>
operator<sensors(rest)|channels(kernel) → cortex(kernel); real → vector3; inverse kernel>
operator<transition head: device:run1 → head>
```

The grammar is the printer's output (a class, `<`, `;`-separated members, `>`), with `×` for products, `|` for a restriction and `M`, `T` (single
capitals) for variables. It is small enough to parse by hand: no expression, no arithmetic.

## 4. Generics: signatures with variables, solved by unification

A node declares a **signature**: its ports' types written with manifold variables, and its **move** (§5). The signature is declarative; most of
today's `plan` disappears into it. What the signature cannot say (a parameter that picks a band, a level, a form) is in a small **`refine`**
function, which receives the unified inputs and the parameters and may only narrow the outputs or refuse.

```cpp
// C++ builder: checked by the compiler, printed by the printer
NodeSpec n = node("band.analytic", Kind::chunk);
auto M = var("M"); auto T = time_var("T");                         // T: role time (a stream's samples)
n.signature = sig(Move::pointwise_linear)
  .in("x", field({M, T}).value(real))
  .out("y", like("x").value(complex));                               // like(x): x's type with the changes listed
n.refine = [](Unified& u, const Json& p) { if (!p["band"].is_null()) u.out("y").res.band = band_of(p); };

n = node("kernel.apply", Kind::chunk);
auto A = var("A"), B = var("B"), R = var_rest("R");                 // R: the other factors, carried through unchanged
n.signature = sig(Move::apply)
  .in("x", field({A, R}).linear())
  .in("kernel", operator_(from_or_restriction_of(A), B))
  .in("tiles", selection_ladder(B).optional())
  .out("y", like("x").on({B, R}).value(target_value("kernel")));
```

**Solving.** `check` walks the uses in order. For each use it makes fresh variables, **unifies** each input's type with the declared pattern
(binding `M`, `T`, `A`, `B` to the terms they meet), instantiates the outputs, runs `refine`, and records the outputs' types. A variable bound to
`cortex(kernel)` in one port and met as `glued(cortex(kernel))` in another is a refusal, naming both. So a chain carries its manifolds through:
the output of `kernel.apply` is known to lie on the kernel's target before any data, and the `measure` after it is checked against the joint
partition's surface on that term, not on a label.

**Unification rules** (the only ones):
- a variable binds to any term of its kind constraint (a `time_var` only to a factor whose role is time; `frames_var` to time or frames);
- `id` = `id` when the ids are equal; `anonymous` = `anonymous` only when the same number (or declared the same, §10);
- `members(P)` = `members(Q)` when P and Q are the same selection; `restricted(M, S)` = `restricted(N, U)` when M = N and S = U;
- `derived(rel, M)` = `derived(rel, N)` when M = N (in `run`, both resolve to the stored id first);
- the one subsumption, at `apply` only: an operator from `restricted(M, S)` accepts a field on `M` (the kernel reads its channels of the
  recording: today's padding to every member of the field's first factor) and a field on `restricted(M, S)`; a field on `restricted(M, U)` with
  U ≠ S is refused (the channels of a cross-spectrum that are not the kernel's).

**Nodes that make manifolds** declare them as outputs or as parts of outputs: `made(line, Role::frequency, cs{"frequency","hz","Hz"})` (Welch's
frequency line), `made(points, Role::modes, cs{"spectrum:<op>","eigenvalue",…})` (a mode axis), `members(sel)` (a frame line, a set of time
tiles). A made manifold is a fresh term keyed by the use and the output: two outputs of one use share it, two uses make two. That the runner later
finds an identical stored line and reuses it (`Unit::line`, `2026-11-17-definition-outputs.md`) is storage, not typing. A node that makes a
selection (`spans.detect`, `cores.detect`, `tiles.make`, `joint.make`) gives it an identity `"<use>.<output>"`, so the measurements over its
members name it.

## 5. The moves and their typing rules

Every node is declared as one move. The move fixes what the node may do to the **domain**, and that is where the cardinal rule lives.

| move | signature (schematic) | the rule the checker applies |
|---|---|---|
| **pointwise** | field<D; V> (+ constants) → field<D; V′> | the domain is **unchanged** (the same factors, the same cells). Linear (`band.analytic`, `band.filter`) keeps `nonlinear`; non-linear (`field.magnitude`, `field.power`, `band.envelope`, `phase.current`) sets it. A constant input (a statistic: `relative`'s baseline) must be **of** the field: its `of` domain equals D without the streamed factor, and its resolution and quantity are the field's |
| **apply** | operator<A → B>, field<A × R> → field<B × R> | **a change of basis is an apply** (FFT, wavelet, eigenmode projection) of an operator the node itself defines, and the value it makes lies on the line that operator makes: `fft.spectrum`'s output is on `channels × frequency`, a product whose second factor is the line the node makes (keyed `fft.spectrum.spectrum`; bin k at k × rate / n Hz, origin 0, unit Hz; n carried by the block), never an untyped block axis; a later node reads the bands on that line. The field is linear (D106); A by the subsumption rule; cells: the operator's from-cells are the field's; value type: the operator's target value type, complex when the field is; gauge: the operator's target gauge (G5). The operator acts on one declared factor: the first (a kernel), or the streamed one (`tiling.frames`: the interpolation from the time line onto `members(tiling)`) |
| **compose** | operator<A → B>, operator<B → C> → operator<A → C> | ends meet (manifold, cells, value type); a pair (`inverts`) has its ends swapped: the Studio's chain and pair rules, D106 |
| **restrict** | field<M × R>, selection<indices or a level on M> → field<restricted(M, S) × R> | the selection is on M; `at_level` restricts the stack of levels (a component restriction: the domain is unchanged, `res.levels` narrows) |
| **measure** | field<D>, selection<S on F ⊆ D> → measurement<D with F replaced by members(S)> | **the only move that removes a factor.** S is any selection on factors of the field, in order; no property is needed, and one region or time range is a selection of one member. Over a **PRODUCT GRID** (a `set` made by a rule per axis, of grids only: Studio D100 `productSelection`, mixed radix, first factor fastest; D129's joint partition of space × time) the measurement removes **every factor the grid covers** and gains one member set (`members(grid)`, label `joint`). The levels of a product grid are per axis or **coupled**: a frequency band sets the time level (constant-Q `natural_level`) and an eigenvalue band sets the dyadic spatial level, so the field's band IS the grid's level on that axis and a mismatch is refused (a single band selects: it is no frequency axis; resolving frequency within it is a transform, an `apply`). An **irregular** selection (detected events, scouts, an atlas) is measured **member by member**: its members may overlap and have no levels, and nothing is rolled up. A ladder is accepted with its level chosen by the field's resolution (the level IS the band), and refused when no level resolves it. A joint partition's (a cells selection's) resolution must equal the field's (band and level). The measure must fit the quantity (the node's table: `amplitude` of `amplitude` or `envelope`, `power` of `power`, `phase_current` of `phase_current`). `exclude` (spans on the streamed factor) leaves members out, never a factor. A selection on the streamed factor needs an `accumulate` node; one on another factor keeps the stream (pointwise in time). Disjointness is asked of nothing here: only a grid has a hierarchy to roll up through (`summarize`). `measure_events` (the tile measurements of each event, event by event) takes the overlapping events `detect` makes |
| **summarize** | measurement<… × members(P) × …> → statistic<axes: the rest; of: the measurement> | across a selection's members: the statistic remembers what it is of (`OfType`), which is what a later node compares. **Roll-ups go only through a grid's closed-form hierarchy** (a coarser level of a dyadic tiling is the shifted code of the finer, never a parent-map array). Across an **irregular** selection's members (the bursts of a run) the statistic is a **`summary` statistic** of their measurements (count, mean, median, a histogram): the same node, no hierarchy, overlap allowed. Statistic **estimation** (Welch, a cross-spectrum, a covariance, a histogram, the spatial baseline) is `measure` over a declared partition of the stream (the segments, the good frames) followed by `summarize`, fused in one accumulate node that declares the partition it reduces over |
| **detect** | field<D> or measurement<… × members(P)> → selection<on D, or on the factor P partitions> (+ measurements over its members) | the input is a scalar field or measurement; the made selection is an **irregular** one (`partition` false: sparse, any extent, may overlap, no levels, not indexed by the grid; a lookup only on request), with an identity and `detected_over = P`, which is provenance (the grid whose members bound the event: a detected member's bounds come from that grid; each member is described by centre and extent); the node's other outputs are measurements over `members(made)` (an event table: frame, centre, area, peak). A detection over events (`cores.tracks`) makes a set over `members(events)`, itself irregular |
| **make** | (parameters, manifolds) → selection or manifold | a partition, a ladder or a line made from geometry or parameters (`tiles`, `constantq`, `frametiling`, `joint` as the product of two partitions, `eigenmodes`' mode axis). No field input |

**The cardinal rule, in the structure.** The checker refuses, for every use: a `pointwise`, `apply`, `restrict` or `compose` node whose output
domain lacks a factor of its input's (other than the one an `apply` maps); a node that removes a factor without being `measure` or `summarize`;
a `measure` with no selection input and no run span; a `summarize` of a field (a field is measured first). Nothing a node's own code does can get around this,
because the output types are made from the signature, not from the node.

**Event tables.** `cores`, `patches` and `tracks` return a selection and many measurements over its members. The inventory found no other shape,
so the rule above (detect with measurements over `members(made)`) covers every event finder, and the per-event outputs' types are known before data.

## 6. The checker

`check` (`graph.h`) keeps its order (structure, nodes, traits, kinds by section, arrows, halo, retiming, outputs) and its arrow step becomes one
shared function, `type_use(use, inputs, signature)`: unify, the move's rule, `refine`. A node without a signature (until its PR converts it) runs
its old `plan`. Refusals name the definition, the section and index, the node, the output, the port, the two types in the printed form, the rule,
and what to do. Examples, each from a real mistake (the first five are in the prototype, appendix B):

```
global_bursts: chunks[3] measure.over_partition -> cell_amplitude: port `over`:
  the joint partition `cells` is at level 5 of `tiles`; x is band-limited to level 4 of `tiles`
  (x = field<cortex(kernel) × time(rest); real; amplitude; band 8-16 Hz; level tiles@4>, from chunks[1] kernel.apply).
  A measure over a joint partition takes a field at the partition's own levels (the level IS the band): set kernel.apply's `level` to 5, or find the
  cells of level 4.

cell_phase_current: chunks[2] kernel.apply -> sources: port `x`: x is non-linear (made by chunks[1] field.power: power).
  A linear map reads the chain before any non-linear node (D106): apply the kernel first.

band_bursts: after[4] spans.detect -> bursts: port `x`: x is field<sensors(rest) × time(rest); real; envelope; …>; events are detected on a
  measurement over a time partition's members. Measure it over the constant-Q level first (measure.over_partition with `over: ladder`).

vortex_null: chunks[3] kernel.apply -> curl_frames: port `kernel`: the kernel reads sensors(rest)|channels(kernel); x lies on
  sensors(rest)|channels(cross_spectrum) (the surrogate is drawn on the cross-spectrum's channels). Make the cross-spectrum with
  `--channels-of <kernel>`.

phase_gradient (a converted verb): port `frames`: x is spin 1 in gauge geometry-central@8579865/vertex:<cortex>:<fingerprint>; the operator was
  built in gauge …:<glued>:<fingerprint>. Insert frame.rotate, or recompute the stored field (it predates the gauge record).

headmodel (a converted verb): port `channels`: positions in head/device:run1 (m); the cortex is in head/scs (m). No transition device:run1 → scs is
  stored for this acquisition; import the recording's device → head transform (it is a transition operator), then run again.
```

**Before any data** means before any array is read. In `run`, the inputs' types come from rows (§8); in `compute`, from the arrays' metadata (§10).
`--dry-run` prints each value's type (`Checked::text()` gains a column).

## 7. How a definition declares types

**Most are inferred.** A definition's inputs already say their class and their relations (`manifold: kernel.to`, `from: recording.manifold.0`);
with the rows found, that gives every input's type. Every step's outputs come from its signature. The `outputs:` kinds are checked against what
the graph makes (a statistic declared as a measurement is refused). Annotations are needed only where the relation is not in the definition today
and a stored row could be the wrong one:

- `on:` on a selection that lies on a **product** (a joint partition), `[LINK, LINK]`: the finder requires the selection's manifold to be that
  product (today the cells are found by `params.joint.*` alone, and a joint partition of another surface is refused only inside `measure`);
- `selection: tiling` for a frame tiling (a closed-form spans Selection that tiles its line: a partition); `selection: spans` stays the type of
  detected or stored events;
- `of:` on a statistic input, a LINK to what it is a statistic of (the spatial baseline's surface, a cross-spectrum's channels), checked against the
  stored axes;
- `type:` on a composite's ports (the printed form, §3.3), since a composite has no rows to infer from.

**Before / after, three shipped definitions.**

`band_bursts` (inputs: a recording, the constant-Q ladder of its time line, the bad spans). **No change.** Every value is inferred; `--dry-run`
now prints, among others:

```
amp        field<sensors(rest) × time(rest); real; envelope; band 8-16 Hz; non-linear>
amplitude  measurement<sensors(rest) × members(ladder@8-16 Hz); amplitude>
baseline   statistic<summaries [median, mad]; axes sensors(rest); of amplitude>
bursts     selection<spans on time(rest); partition; detected over ladder@8-16 Hz>
burst_amplitude  measurement<sensors(rest) × members(bursts); amplitude>
```

`global_bursts`: the joint cells gain their manifolds.

```yaml
# before
  cells:     {selection: set, where: {params.joint.space_level: $space_level, params.joint.band: $band, params.joint.recording: "{recording}"}}
# after
  cells:     {selection: set, on: [kernel.to, recording.time], where: {params.joint.space_level: $space_level, params.joint.band: $band}}
```

The `params.joint.recording` filter goes: the product with `recording.time` already says which recording. The cells' resolution (band, level) is read
from the row (§8) and compared in `measure`.

`spatial_events`: the frame tiling and the baseline are typed.

```yaml
# before
  tiling:    {selection: spans, manifold: recording.time, where: {name: $tiling}}
  baseline:  {statistic: mean, where: {name: $baseline_statistic}}
# after
  tiling:    {selection: tiling, manifold: recording.time, where: {name: $tiling}}
  baseline:  {statistic: mean, of: kernel.to, where: {name: $baseline_statistic}}
```

`relative` then checks that the baseline is of the same chain (quantity `power`, band, levels, the glued surface, the same frame tiling's members)
from types, where today it reads the statistic's stored chain in `prepare`.

## 8. What the runner checks against stored data at plan time

The runner turns each found row into a `PortType` through io's task API (`describe`, extended to return the type's parts; no SQL outside io):

| class | from the row (and its links) |
|---|---|
| field | the manifold's factors (`geometry_dimension.manifold_id` of a product), each factor's kind, n, role (`time` for a line in `time/clock`), coordinate system (`geometry.coordinate_system_id` → space, name, unit), spacing; `cell`, `value_type`, `n_components`, `unit`, `kind` → quantity; the producer record's `type` (below) for resolution and linearity; the gauge record (G5) |
| operator | `from_manifold_id`/`to_manifold_id` (and `from_selection_id` → a restriction), cells, value types, `kind`; `from/to_coordinate_system_id` → a transition; the gauge record |
| selection | `type`, `manifold_id` (or a space), `cell`, a dictionary (a partition), the ladder's members with `resolves_hz` or the Weyl band (the level's resolution), a joint partition's levels and band |
| statistic | `axes_json` (each axis a manifold, a selection or named entries), `measure`, `of_field_id`/`of_statistic_id`, `partition_selection_id`, the producer record's `type` |
| manifold | itself |

**Types round-trip through storage.** When the runner stores an output, it writes the output's printed type into the row's producer record (`"type":
"field<…>"`, with ids for the terms). A stored result read back as an input then has exactly the type it was made with: the band, the level, the
quantity, linearity, what a statistic is of. A row stored before this (no `type`) gets the type its columns give. A node that needs a fact the
columns do not hold refuses with the reason and the run that recomputes it (development data may be recomputed).

**Stored-data checks that stay at plan time** (they read rows, never arrays): an operator's rows and columns against its manifolds' vertex counts
(the f-vector is in the row); a kernel's `from_selection` stored; a ladder's level present; stored events snapping to the tiles of the partition
they are measured over (they read `selection_span` rows, small); the gauge fingerprint and the old-gauge refusal (G5); a transition stored between
two coordinate systems a node needs. A check that needs an array's values (finite values, a non-empty bin) stays in the node's `prepare` or run.

## 9. The gauge and coordinate systems

**The gauge (G5)** is part of a field's and an operator's type exactly as `2026-11-17-geometry-central-switch.md` §5.2 says: `spin` and `gauge`.
Signatures declare per port either a fixed spin and "same gauge as input X" (covariant: most tangent nodes) or a produced gauge (a node that builds
the operators: the bound surface's). Unification treats the gauge as an equality constraint: an arrow whose ends differ in spin, or have the same
nonzero spin and different gauges, is refused naming `frame.rotate`. A `vector3` value is gauge-free, so fields stored embedded (the phase current,
the optical-flow velocity) carry no gauge. **Frames** (canonical north / west, anatomical) are not gauges: a node that reports in a frame takes the
frames field as an input, typed `field<surface; matrix3; canonical frames>`, and its output records the frame in its quantity.

**Coordinate systems** are part of a manifold's type (`Manifold::cs`), so two sensor manifolds in different frames (device and head) are different
types, as the audit asks. A node that reads **positions** (a head model, a nearest-vertex lookup, a sensor-to-cortex projection) declares that its
manifold inputs share one coordinate system, or names the one it needs (`scs`). The checker refuses when they differ, naming the transition that
would bring one into the other and whether it is stored. In this spec the checker does **not** insert transitions by itself (question 3): a
definition applies a transition explicitly (`apply` of an operator of kind transition to a manifold's positions is the one `apply` whose result is
a manifold: the same manifold term with another `cs`).

Two dependencies: the checks are only as true as the rows. Until gap **G1** of the audit is closed (every sensor manifold written under `scs`
whatever its frame), a coordinate-system check on sensors passes on wrong data. G1 is fixed before the check is turned on for positions (PR T5).
Time lines and frequency lines are typed by their space and system (`time/clock`, `frequency/hz`): two recordings' lines are of one space, hence
comparable, and are still two manifolds (two identities), so a node that pairs two recordings (`repeat`, `null`) states which partition they are
compared over.

## 10. `compute`, the bindings and untyped arrays

`compute` (`2026-11-17-compute-and-run.md` §6.1, §7) hands arrays to a node or a graph. Each array is given an **anonymous** type from its shape
and the caller's metadata:

- a field array `[rows × cols × comps]` gets `field<#1 × #2; V>`, `#1`, `#2` fresh anonymous manifolds with `n = rows, cols`; a streamed input's
  last factor has role time and the caller's sample rate; the value type from the array (`complex` when it has an imaginary part) or the metadata;
- **labels** (today's `on` / `to` strings) become the names of anonymous manifolds: two arrays labelled `on=["sensors", "time"]` lie on the **same**
  anonymous sensors and time line; an operator labelled `from="sensors", to="cortex"` reads them. Unlabelled arrays never unify with each other
  except through a node's variables (a chain `analytic → magnitude` needs no label: both are the variable `M`);
- a partition is an integer member index per element (`-1` for none) with a label naming the factor it partitions (`compute-and-run.md` §2); it
  types as `selection<set on #k; partition>`;
- the gauge: a complex tangent array without a gauge in its metadata is refused (G5 §5.2); `vector3` needs none;
- a file's sidecar holds each array's printed type, so `compute` on files reads its types back exactly.

The check is the same function as in `run`; only the input types' origin differs. Anonymous manifolds still catch the internal mistakes of a chain
(a kernel whose columns are not the field's rows, a linear map after a non-linear node, a measure over a partition of another manifold); what they
cannot know (whether `#1` is the cortex the caller thinks) is the caller's, and the printed types make it visible. The Python and MATLAB faces
accept the labels as keyword arguments (`on=`, `to=`) and add nothing else.

## 11. The PR plan

The point is that **D1 part 2** (the runner storing declared outputs, and `psd` as YAML) and the **~35 verb conversions** are written once, on the
types, rather than written on `finder::Value` and `facts` and then rewritten.

| # | PR | contents | waits for |
|---|---|---|---|
| T0 | this spec | the inventory script (`tools/type_inventory.py`), the design, the plan | — |
| T1 | the types | `nxr::graph`: `Manifold`, `Domain`, `Resolution`, `SelectionShape`, `OfType`, the extended `PortType` (G5's `spin` and `gauge` included if G5 has not landed; G5 then only adds `frame.*` nodes), the printer and parser, `unify`; additive: `check` unchanged | C1 (merged) |
| T2 | signatures and the shared checker | `NodeSpec::signature`, `Move`, `refine`; `check`'s arrow step runs `type_use` for nodes with a signature, their old `plan` otherwise; the move rules and the cardinal-rule refusals; the 31 steps the definitions use given signatures (their type checks leave `plan`; parameter checks stay); `facts` no longer read across nodes | C5 (the steps are `NodeSpec`s in `pipelines/graph/nodes/`) |
| T3 | the runner's types | input types from rows (task API `describe` returns them), the producer record's `type` written and read, the outputs' declared kinds checked, the YAML annotations of §7 (`on:`, `of:`, `selection: tiling`), `--dry-run` prints types; the shipped definitions edited (§7) | T2, C4 |
| D1.2 | the runner stores declared outputs | `2026-11-17-definition-outputs.md` D1's runner half (`store_output` per kind), on T3's types: a declared axis `on: LINK` is a domain factor, a `line:` axis a made manifold, `partition:` a selection the node declared | T3 |
| D2–D3 | `psd` and `rotary_psd` as definitions | `welch.psd` declared `measure` (segments) + `summarize`, its frequency line `made`; unchanged otherwise | D1.2 |
| T4 | `compute`'s types | anonymous manifolds and labels, partitions as member arrays, the sidecar types, the bindings' `on=`/`to=` | T2, C8 |
| T5 | coordinate systems and transitions | `Manifold::cs` read from rows; the position nodes' constraint; the transition operator's type; G1 fixed first (importers) | T3, audit G1 and G2 |
| V1… | the verb conversions | each verb a definition on typed nodes, in the order of `definition-outputs.md`: spectra (`sourcepsd`, `crossspectrum`, `sourcetensor`, `quadrature`), then geometry makers (`tiles`, `constantq`, `frametiling`, `joint`, `eigenmodes`: `make` nodes), then source imaging (`noisecov` as a statistic, `inverse`, `headmodel`, `pointspread`), then flow and patterns, then nulls | D2–D3, and T5 for `headmodel` |

Each verb conversion then writes, once: its nodes' signatures (the move and the ports), its definition (inputs with relations, outputs with kinds),
and tests against the references. It never writes a type check by hand: the inventory's nine recurring mistakes (appendix A.3) are all rules of §5
or plan-time checks of §8.

## 12. Tests (scoped, through `tools/tests.py`; synthetic data only)

- **`test_graph_types`** (T1): printing and parsing round-trip for every class; unification: variables, ids, anonymous, members, restriction,
  derived, the `apply` subsumption, and each refusal's message.
- **`test_graph_moves`** (T2): one tiny graph per move and rule of §5 that checks, and one that is refused, with the message checked: the
  cardinal-rule refusals (a pointwise node that drops a factor, a `measure` with no partition, a `summarize` of a field), D106, the gauge.
- **`test_definitions_typed`** (T2, T3): every shipped definition checks on synthetic rows; its `--dry-run` types are compared with a golden text;
  the **seeded mistakes** of appendix B (each an edit of a shipped definition) are refused with the expected rule. The prototype's list is the
  starting list; a new real mistake found later is added there first.
- **`test_types_roundtrip`** (T3): an output stored and found again as an input has the type it was made with; a row without a `type` gets its
  columns' type; a node needing a missing fact refuses with the rerun.
- **`test_no_cross_facts`** (T2): no node reads a `facts` key it does not write (a source scan, as the layering guards).
- **`test_compute_types`** (T4): the same graph checked from arrays with labels and without; a gauge-less complex tangent array refused.
- `tools/type_inventory.py --section mistakes` stays green at every PR (all seeded mistakes caught; `--section types`: every definition checks).

## 13. Open questions, each with a recommendation

| # | question | recommendation |
|---|---|---|
| 1 | Signatures in C++ (a builder) or in text (the printed form, parsed) | **C++ builder for registered nodes, text for YAML** (composites, sidecars). The compiler checks the builder; the parser serves only what has no C++ |
| 2 | Are `members(P)` and a frame line stored as manifolds | **No.** A member set is the partition's (dictionary and count); a measurement's rows are keyed by member. A frame line is stored only when a field on it is materialized (the outputs spec's `manifold` kind, `type: line`), then found and reused as today's lines are |
| 3 | Should the checker insert a transition or an implicit restriction by itself | **Not in this spec.** The one implicit move is the kernel reading its channels of a field on the whole sensor manifold (today's behaviour, and the operator's own `from_selection`). Transitions are explicit nodes; an automatic composition can come later, once G1/G2 make the rows true |
| 4 | Quantity names: free strings or a closed vocabulary | **A closed vocabulary per class in `docs/reference.md`** (the stored `kind`s and the measure registry already are one); a node naming a quantity outside it is refused at registration |
| 5 | Units: compared or carried | **Carried, compared only where a node asks for equality** (a ratio's ingredients, a speed's length unit). No algebra: today's `(x)^2` strings stay as written by the nodes that make them |
| 6 | The stack of spatial levels: components or a domain factor | **Components** (`res.levels`), as today's arrays hold them: `at_level` is a restriction of the components, `scale` a pointwise function across them. A level stack is neither time nor space, so treating it as components keeps the cardinal rule simple |
| 7 | A partition of the stream inside a chunk node (`tile_max`: tiles per sample) | **Allowed**: a `measure` over a partition of a non-streamed factor keeps the stream and is pointwise in time; only a partition of the streamed factor needs an `accumulate` node. `tile_max` becomes `measure` (mean over tiles) then a summary across members (max), declared as two moves of one composite |
| 8 | When a signature and the old `plan` disagree during T2 | **The signature wins and the old check is deleted** in the same PR, with a seeded test for it; a node is never checked twice |
| 9 | Types of hand-wired verbs before their conversion | **None.** A verb is never extended (`ARCHITECTURE.md`, Transition); it gets types when it becomes a definition |

## 14. The Studio, compared (read-only)

The Cortical Flow Studio shows that such checks work, but it checks for different needs, and nxr-neuro does not copy its classes.
- The Studio checks **stored rows after the fact** (`backend/src/db/pairs.ts`: an inverse pair's ends swapped, a composition's factors meeting, a
  field made by an operator from an input on the operator's manifold or a factor of it). nxr-neuro checks a **graph before it runs**; the pair and
  chain rules are kept as the `compose` move's rule.
- The Studio's field types are an **algebra** (ℝ, ℂ, ℍ: `backend/src/valueType.ts`) separate from storage, and a ℂ layer is stored embedded as a
  `vector3` with the algebra in its producer record. nxr-neuro keeps the schema's value types as the type and stores complex tangent values with a
  gauge record (G5); the algebra is a printing of the value type.
- The Studio's GPU chains have a fixed order of stages (members → filter → transform → forms → measures) and no generic signatures; its
  manifolds are the scene's, so it needs no manifold identity in a type. nxr-neuro's graphs are arbitrary, run on many subjects and are composed by
  users, so the manifold term, unification and the move rules are its own.
- Neither has a typed partition: the Studio's `measureOverPartition` takes whatever selection it is handed. Here the partition is the only door
  to a reduction, by type.

## Checklist (the architecture policy)

- **Each new compute function callable with a matrix from anywhere?** Yes: types are metadata the executor checks; nodes still receive plain
  arrays, and `compute` gives arrays anonymous types.
- **The same chain on another input by naming it?** Yes: signatures are generic in their manifolds; a definition's relations name inputs, never
  manifolds by name.
- **Every loop over samples or chunks, and every file access, inside io?** Yes: the checker reads types only; the runner reads rows through the task
  API to make them.
- **Every choice of data made by the caller?** Yes: the type system refuses or prints; it never picks an input, inserts a transition or chooses a
  level.
- Reductions only as measurements over partitions: enforced by the move rules (§5). Precision follows the input: unchanged (`NodeTraits`), not a
  type.

---

## Appendix A. The inventory

### A.1 How it is made

`tools/type_inventory.py` reads the tracked sources and prints four sections; rerun it after each PR of §11:

```
tools/type_inventory.py --section ports      # every definition's inputs, uses and outputs; every registered step's ports
tools/type_inventory.py --section checks     # today's per-step checks, by what they check
tools/type_inventory.py --section types      # the rules of §5 run on every shipped definition: each value's type
tools/type_inventory.py --section mistakes   # appendix B
```

The `types` section is the spec's executable sketch in Python (the implementation is C++). It assumes, where a definition does not relate an input,
what the found row would say; the assumptions are printed with the section. Today: **15 of 15 shipped definitions check; the 31 steps they use
have a rule; their moves are pointwise 9, detect 7, measure 6, statistic estimation 5, apply 2, summary 1, restrict 1.** The connectome steps
(11, used by no shipped definition) are not yet given rules.

### A.2 Today's per-step checks (generated, `--section checks`)

```
282 checks in 39 steps
    2  linearity (D106)
   35  class or producer of an input          e.g. "tile_crossings: events must be the spans of the `cores` step"
   13  value type                             e.g. "envelope: x is complex; the envelope is taken of the real field"
   37  manifold identity                      e.g. "frames: the frame tiling … is of another time line than x's"
   54  partition, ladder, level               e.g. "tile_crossings: the ladder … has no level …"
   13  band, rate                             e.g. "frames: the tiling was made for … samples a second, x has …"
    4  stored record                          e.g. "relative: … records no chain: it is not a `baseline`'s"
  124  parameter or array shape               e.g. "nearest_vertex: points and positions are [n x 3]"
```

The classification is by keywords and is approximate; read it as an order of magnitude. The first seven rows (158) are what §5 and §8 replace.

### A.3 The hand-wired verbs (by reading the code; not generated)

What each verb's inputs and outputs are in the model's terms, and the move each internal step is. "made" marks a manifold or selection the verb
creates. Common to every recording verb: the recording is a field on sensors × time (points × a `time/clock` line); the kernel an operator from the
sensors restricted to its `from_selection` to the cortex (`scalar` → `scalar` fixed, → `vector3` free); bad spans a spans selection on the time line.

| verb | inputs → outputs | moves | checks it makes today that become type rules |
|---|---|---|---|
| `noisecov` | recording → covariance (today an operator; to be a **statistic**, audit §1.0) | restrict (good channels of types), measure over the span + summarize | the channel types |
| `inverse` | forward operator, covariance, sensor-type partition → inverse kernel operator, whitener operator | make (an operator) | covariance on the forward's channels; its `from_selection` stored |
| `headmodel` | sensor points (cs), cortex surface (cs), head surfaces → forward operator | make | **the only coordinate-system check today** (cortex and channels in one system) |
| `pointspread` | forward, kernel, tile ladder → statistic (named-entry axes) | apply, apply (spatial band), detect, measure | the kernel reads the forward's sources |
| `eigenmodes` | surface, atlas partition → made: mode axis (`spectrum:<op>/eigenvalue`), glued surface, intrinsic complex (cells 0, 1, 2); eigenvector operators, exterior derivatives (cells change) | make | the surface has one or two pieces; the atlas fits it |
| `frames` | surface, its sphere → canonical frames field (`matrix3`), angles, singular-vertex selections | pointwise, detect | sphere's vertices = surface's |
| `tiles` | surface → made: tile ladder (levels: partitions, each with its spatial band) | make | — |
| `constantq` | a time line → made: constant-Q ladder (levels: tilings with `resolves_hz`) | make | the line is closed-form |
| `frametiling` | recording's time line → made: frame tiling (a tiling of the line) | make | the line has a rate |
| `joint` | tile level, constant-Q level → made: a partition of surface × time (product) | make (product of partitions) | the levels exist; the code fits 32 bits |
| `lowpass` … `notch` | recording → recording on the same domain | pointwise (linear, windowed) | the band below Nyquist |
| `psd` | recording, bad spans → made: frequency line, segments tiling; statistic on sensors × frequency | measure (segments) + summarize | span ≥ a segment; a whole segment free of bad spans |
| `sourcepsd` | recording, kernel → statistic on cortex × named bands | apply, measure + summarize, measure (bins → bands) | kernel on the recording's channels; rows = 3 × vertices |
| `sourcetensor` | recording, free kernel, canonical frames → statistics (`hermitian3`, `quaternion` Stokes in the frames) | apply, measure + summarize, pointwise | a fixed kernel refused; frames present |
| `crossspectrum` | recording (channels: types, or a kernel's) → statistic on channels × channels (complex), per band or per window | restrict, measure + summarize | the operator's channels in the recording |
| `quadrature` | recording, kernel → statistics on cortex × bands | apply, measure + summarize, pointwise | kernel / channels / vertices |
| `hodgepower` | recording, free kernel, mode package → statistic on mode axis × bands | apply (kernel then modes: compose), measure + summarize | a fixed kernel refused; the package fits the kernel's surface (found by name today) |
| `speedspectrum` | as `hodgepower` → made: index lines and products; statistics | apply, measure + summarize | as `hodgepower` |
| `phasegradient` | cross-spectrum statistic, free kernel, tile leaves, frames → measurements over the leaves | apply, pointwise (tangent), measure | kernel on the statistic's channels; surface unit a length; frames present |
| `cyclepattern` | window cross-spectrum series, free kernel, package → statistics (complex; glued surface × patterns) | apply, measure + summarize (an eigendecomposition across windows) | kernel on the series' channels; a finer level refused |
| `opticalflow` | recording, kernel, frames, leaves, frame tiling, package → made: time line, product; fields; measurements over leaves × frames | apply, pointwise, measure | kernel / vertices; at least 8 frames |
| `cellquadrature` | window cross-spectrum, joint partition, kernel, tile level, bursts → measurements over cells and bursts | apply, measure | **the cross-spectrum's tile level = the joint's time level; the kernel's channels = the statistic's; the kernel's surface = the joint's** |
| `patches` | a scalar map on a surface, tile level → band-passed field, patches selection, measurements, statistics | apply (spatial band), summarize, detect, measure | a scalar map, one value per vertex, on a surface |
| `scaleselection` | joint partitions, labellings, cell measurements, baselines (made by a definition) → events and measurements | detect, measure | the joints on one surface; shapes [tiles × time tiles] |
| `pyramid` | recording, kernel, tile leaves, frame tiling, bands → made: product, joint partition; measurements | apply, measure | frames present; kernel / vertices; anchoring |
| `measure` | fields, a partition → measurements | measure | the partition on a surface, or a contiguous tiling of time |
| `statistic` | a measurement over a partition → statistic | summarize | the measurement exists over that partition |
| `wishart` | recording, window series, bursts, PSD → statistic | summarize (draws) | the bursts on the recording's time line; the PSD's shape |
| `spatialnull`, `null`, `repeat` | stored statistics or measurements of a definition's run → statistics (runs a definition per draw) | summarize | the two runs over the same tile level; shapes |
| `reduce` | many subjects' measurements → the template's measurements | measure over the subjects | members share leaves and bands (cross-subject: out of scope) |

**The nine recurring mistakes** the verbs guard against, most frequent first, and where each goes: (1) a kernel on other channels than the
recording or statistic: the restriction rule at `apply` (§4); (2) a kernel's rows against its surface's vertices: a plan-time row check (§8);
(3) a fixed kernel where a free one is needed: the value type (`scalar` vs `vector3`) at `apply`'s target; (4) a missing prerequisite (frames,
ladder, package, joint): the finder, with the run that makes it; (5) a level or partition mismatch: `measure`'s resolution rule; (6) a selection on
another time line, a span shorter than a segment: unification of the time factor, and a plan-time row check; (7) coordinate systems and units: §9;
(8) value types and shapes: the signature; (9) overwrite and ambiguity: the finder and the cache, not types. Downstream nodes found **by naming
convention** (`<surface>_sphere`, `hodgeDirac__<surface>`, `<surface> tiles`) become relations a definition declares (`manifold: kernel.to`), as
the shipped definitions already do.

## Appendix B. Seeded mistakes (generated, `--section mistakes`)

Each is a small edit of a shipped definition; each is refused by the rules of §5 before any data is read.

| definition | the edit | refused by |
|---|---|---|
| `band_bursts` | measure the analytic signal (complex) instead of its envelope | `measure`: x is complex |
| `cell_phase_current` | the kernel applied after `power` | `apply`: x is non-linear (D106) |
| `global_bursts` | the spatial band at level 4, the joint cells at level 5 | `measure`: the partition's level against x's |
| `cortical_cells` | the time band changed to 14–30 Hz, the cells of the 8–16 Hz level | `measure`: the partition's band against x's |
| `spatial_events` | frames interpolated after `power` | `apply` (frames): x is non-linear |
| `band_bursts` | events detected on the envelope field, not on its measurement | `detect`: x must be a measurement over a time partition |
| `global_bursts` | a streamed field summarized | `summarize`: a field is measured first (the cardinal rule) |
| `global_bursts` | the tile ladder of the sensors instead of the cortex | `apply`: the ladder is not on the kernel's target |
| `global_bursts` | an output declared a measurement that is a statistic | the outputs' kinds |
| `band_bursts` | `measure_events` of events detected on another band's tiles | `measure` over the events (irregular): the grid they were detected on against x's |

10 of 10 caught.
