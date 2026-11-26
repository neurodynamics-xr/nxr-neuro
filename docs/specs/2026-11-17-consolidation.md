# One selection and measurement system: the consolidation (spec, 2026-11-17)

Status: agreed design, to be implemented before the next build. It governs where it differs from the earlier specs of the round
(`2026-11-17-selection-module.md`, `2026-11-17-type-system.md`, `2026-11-17-pyramid-grid.md`, `2026-11-17-tile-grids.md`, `2026-11-17-flow-verbs.md`,
`2026-11-17-surrogates.md`). Those keep their section 0 rules (five selection types, grids versus irregular selections, closed-form hierarchies,
centre and extent, products as the Studio's D100 and D129, outward rounding); this spec removes the competing definitions that grew around them.

**The rule: no competing definitions.** Every selection is made by one of five maker nodes; every reduction is one of four operations, each with one
implementation; measurements over members live in one table; every measuring pipeline is one `measure` definition fed by a field chain, with named
presets that are only saved parameter sets. Old names are not kept as aliases: a retired name is refused with the name of what replaces it.

Contents: 1 the makers · 2 the operations · 3 quantities, charts and units · 4 field chains, `measure` and its presets · 5 the group scope ·
6 storage and the migration · 7 lines · 8 absorbed items · 9 the mapping table · 10 tests · 11 checklist.

---

## 1. The five selection makers (`select.*`: compute creates, io stores)

Each maker is a node (registered in `pipelines/src/select_nodes.cpp`) whose maths is a compute function (`compute/include/nxr/selection/`). It reads
no data: its inputs are manifolds (a surface, a line) and stored selections; its parameters are QUANTITIES (section 3). Its output is a
`selection<…>` port the next node consumes through a typed edge (never re-made from the same parameters); a definition that stores it writes it once
(get-or-create by the maker's parameters: the cache key is the maker's name, its resolved parameters and its inputs' ids).

Types below are in the printed form of `2026-11-17-type-system.md` §3.3; `S` is a surface manifold variable, `T` a time line, `L` a line.

### 1.1 `select.dyadic_surface`

| port | type |
|---|---|
| in `surface` | `manifold<S; surface>` |
| in `spheres` (optional, `rule: transfer`) | `field<S; vector3; registered sphere>` and the template's `field<sphere positions>` |
| in `template` (optional, `rule: transfer`) | `selection<set on S0; grid dyadic>` (the template's grid) |
| out `grid` | `selection<set on S; grid dyadic; levels 0..D; level attributes {area_m2, diameter_mm, weyl_band_per_m2}>` |

- `rule: subject` (the surface's own tiles; was `tile_grid`, tree subject), `rule: group` (the default subject's grid as the group tree, with the atlas
  definitions; was `group_tree`), `rule: transfer` (a subject's vertices on the group tree through the registered spheres, keeping the template's codes;
  was `group_tile_grid`).
- Parameter `finest`: the finest tile as a QUANTITY (a tile diameter in mm, a tile area in mm², a wavelength in mm, or parts per hemisphere); the
  depth D is resolved from it (section 3.3). The Levi-Civita connection between neighbouring tiles is not part of the grid: it is a measurement over
  the product of a level with itself (section 8), made by `reduce` with measure `levi_civita`.
- Compute: `nxr::selection::dyadic(surface arrays, finest_depth)`, `nxr::selection::transfer(template, match)` (exist), and the level attributes
  (measured mean tile diameter per level; area; the level's Weyl band).

### 1.2 `select.time_grid`

| port | type |
|---|---|
| in `line` | `manifold<T; line time>` (or a field `field<M × T>`, of which only the time line is read) |
| out `grid` | `selection<spans on T; grid time; rule {origin, width, hop} in samples; level k>` |

One node, one rule family: every member is `[origin + i·hop, origin + i·hop + width)` on the line's samples. The variants differ only in how width and
hop are resolved from the stated quantity:

| `rule` | stated quantity | width, hop | was |
|---|---|---|---|
| `tiles` | a duration (s) | `2^k` samples, k = the level whose tile is the first at least the duration (`natural_level`); hop = width | `time_grid(line, k)`; lagfree_envelope `level` |
| `band` | a band (Hz, section 3) | k = `time_level_of_band(lo, hi, sfreq)` (the Studio's `fitLevelOf`: the first `2^k` holding 4 cycles of the band's centre √(lo·hi)) | band_amplitude's per-band grid, joint `time_level` |
| `frames` | a band (Hz) and frames per tile as a power of two (`2^s`) | k_f = `time_level_of_band(band) − s`; hop = width | `frametiling`, `tower::frameLevelFor`, `frame_level` |
| `constant_q` | a frequency range (Hz) and voices | the list of `band` levels of the ladder's bands, each level carrying the Studio's `ConstantQLevelInfo` attributes | `constantq compute` |
| `segments` | a segment duration (s) and an overlap (fraction) | width as Welch's presets resolve it (`tile`, `round`, `even`); hop = width − ⌊width·overlap⌋ (members overlap: a family, not a partition) | Welch's `segmentFor`, declared_outputs "partition:" segments |
| `windows` | a window duration and a step (s) | width and hop snapped to `2^k` samples | cyclepattern's window series |
| `chunks` | none | k = 15 (the chunk grid, `<recording>_chunks`) | the chunk grid |

- ONE time-level rule: `time_level_of_band` (rule 10). `nxr::tower::frameLevelFor` (frames per cycle) and the day-anchored tower origin are retired;
  the origin of every grid is the line's first sample (decision D2 in section 12).
- A grid made by `tiles`, `band` or `frames` is one LEVEL of the line's dyadic time grid: it is not stored as a selection of its own. The line's one
  stored grid row is `<line> time grid` (closed form, all levels); a measurement or a product records the level it is at (section 6.2).
- Compute: `nxr::selection::time_grid(line, k)`, a new `time_rule(line, TimeRule)` for the other variants, `time_level_of_band`, `natural_level`
  (exist in `grid.h`), `constant_q_design` (exists).

### 1.3 `select.bands`

| port | type |
|---|---|
| in `line` | `manifold<L; line frequency>` (the dataset's Frequency line, section 7) or `manifold<L; line eigenvalue>` |
| in `grid` (optional, `rule: levels`) | `selection<set on S; grid dyadic>` |
| out `bands` | `selection<spans on L; grid or irregular; lookup {code → name, lo, hi, unit, chart}>` |

| `rule` | stated | family | was |
|---|---|---|---|
| `frequency` | a list of bands in a frequency chart (Hz, or log10 Hz) with names | irregular (listed), grid when the bands tile an octave rule | `select.bands`, `pyramid.bands`, `constant_q_bands`, the dataset's standard bank |
| `eigenvalue` | spans in the eigenvalue chart (1/m²) or the wavelength chart (mm) | as above | `eigenvalue_bands` |
| `wavelength` | named scales by wavelength (mm) | irregular | `scales_by_wavelength` |
| `parts` | named scales by parts per hemisphere | irregular | `scales_by_parts` |
| `levels` | a dyadic grid's levels (each level's Weyl band) | grid | `pyramid.spatial_bands`, `spatial_bands` |

- A single band SELECTS (it is no frequency axis): a field or measurement of one band carries the band as its resolution. Several bands are a band SET:
  one selection, and the measurement's members include the bands, never a run or a Field per band.
- A consumer takes the set through its `bands` edge and picks a member by code or name; it never re-makes the set from parameters.
- Compute: one header `compute/include/nxr/selection/bands.h` (the five rules), replacing `constant_q_bands` and `eigenvalue_bands`
  (`selection/partition.h`) and `scales_by_*` (`selection/scales.h`, whose named scales stay as the data of rule `wavelength` and `parts`).

### 1.4 `select.product`

| port | type |
|---|---|
| in `factors` (ordered, 2 or more) | `selection<…>` each: a grid level, a band set, an elements grid, or (D100) an irregular set |
| in `bands` (optional) | `selection<spans on L>`: the band set whose members COUPLE levels |
| out `product` | `selection<set on M0 × M1 × …; product; family grid|irregular; radices>` |

- Code mixed radix, FIRST factor fastest (`product_code`, `product_codes`); the joint partition (D129) is `layout: joint`: frames first, radix rounded
  to `2^bits`, code `(tile << bits) | frame`, arrays `[band × tile × frame]`.
- COUPLING. A factor may be stated as a grid with a coupled level: `{grid: <surface grid>, level_of: band}` (the Weyl level of the band's centre,
  `space_level_of_band`) or `{grid: <time line>, level_of: band}` (`time_level_of_band`). With a band SET, the product is made per band member (each band
  its own levels), and the output is a list of products indexed by the band's code. Coupled levels are computed in closed form at plan time and are
  never stored as selections of their own (section 6.2).
- Roll-ups only along grid factors (`rollup_refusal`).
- Compute: `nxr::selection::product`, `joint`, `coarser_code` (exist).
- Retires: `joint.cells`, the `joint` definition, `pyramid`'s joint step, the per-pair `cells` selections found by `params.joint.*`.

### 1.5 `select.events` (irregular selections made by detection)

Four nodes of one family, each a `detect` move (type spec §5): the made selection is irregular (sparse, may overlap, no levels), records
`detected_over` (the grid), and each member is described by centre and extent. Each may also return measurements over its members.

| node | in | out | was |
|---|---|---|---|
| `select.events.spans` | `measurement<members(G) × R>` over a time grid G, a baseline `statistic<of it>` | `selection<spans on T; irregular; detected over G>` | `detect_spans` |
| `select.events.regions` | a measurement over a dyadic level (× a time grid), or a scalar field on S per frame; a threshold statistic | `selection<set on S (× T); irregular>` | `cell_extent`, `patches` |
| `select.events.cores` | `field<S × frames; curl>`, a threshold statistic read by name | `selection<points on S × T; irregular>` | `cores` |
| `select.events.tracks` | `selection<events>` | `selection<set over members(events); irregular>` | `tracks` |

- Filters on members are QUANTITIES (section 3.4): a duration range (s), an area range (mm²), a minimum extent in cells is replaced by a minimum area.
- Refinements keep the family: `select.events.spans` with `rule: keep_where` (members whose measurement passes a stated test: was `fit_gate`) and
  `rule: stages` (onset, plateau, offset of each member from the rise-fall fit: was `burst_stages`).
- `risefall` stays a node: a measurement over each member of an event set (not a maker, not a reduction of a grid).

---

## 2. The four measurement operations

ONE compute implementation each, in a new module `compute/include/nxr/measurement/` (namespace `nxr::measurement`), and ONE node each
(`pipelines/src/measurement_nodes.cpp`). Every other copy is retired (section 9).

### 2.1 The measures (one registry)

`measurement/measures.h`: the additive ingredients `Ingredients` (per row and member: `count`, `weight`, `sum`, `sum_sq`, `max`, `min`, `abs_max`,
and for vectors `sum` per component) and the measures read from them. Every measure names the FORM of the field it is taken of (the Studio's field
forms) and its merge rule:

| measure | of the form | ingredient | merge |
|---|---|---|---|
| `mean`, `sum`, `mean_square`, `max`, `min`, `abs_max`, `count` | the field as it is | sum / sum / sum_sq / max / min / abs_max / count | sum, sum, sum, max, min, max, sum |
| `median` | the field as it is | the held values (exact; capped, as today) | none (never rolled up) |
| `amplitude` | `magnitude` (\|z\|) | sum | sum (mean at read) |
| `envelope_max` | `magnitude` | max | max |
| `power` | `power` (\|z\|²) | sum | sum |
| `phase_current` | `phase_current` (vector) | sum per component | sum |
| `outflow`, `circulation` | a 1-form on edges, a 2-form on faces | sum over the member's barycentric region | sum |
| members' own: `area`, `n_elements`, `duration`, `interval_next`, `centre`, `first_cell`, `last_cell`, `path_length` | none (the member itself) | as defined | per measure |
| `levi_civita` | the surface's frames, on a product of a level with itself | per pair of neighbouring tiles | none |
| `tensor_*`, `gradient_*` | a hermitian3 field in the canonical frames; a scalar field's gradient | sums | sum |

Ratios (amplitude = sum / weight, sd from mean and mean_square, speed, travelling index) are DERIVED at read from the stored ingredients, never
reduced themselves. The registry is data (`io/schema/measures.json`, read by io and handed to compute as a table); the arithmetic is compute's.

### 2.2 `reduce`: a field (or a measurement) over any selection's members

```cpp
// compute/include/nxr/measurement/reduce.h
namespace nxr::measurement {
struct Request { std::vector<std::string> measures; Weighting weighting; };                // weighting: none | given (per element weights)
// fold one block: values [rows × elements] (float or double: precision follows the input), the members as one resolved form
void reduce(Ingredients& acc, const float* values, std::int64_t rows, std::int64_t elements, const selection::MemberOfElement& members, const double* weights, const Request&);
void reduce(Ingredients& acc, const double* values, std::int64_t rows, std::int64_t elements, const selection::MemberOfElement& members, const double* weights, const Request&);
void reduce(Ingredients& acc, const float* values, std::int64_t rows, std::int64_t elements, const selection::Elements& members, const double* weights, const Request&);   // overlapping members
void reduce(Ingredients& acc, const float* values, std::int64_t rows, std::int64_t samples, std::int64_t first_sample, const selection::Rule& tiles, const Request&);         // a time grid, block by block
void reduce(Ingredients& acc, const float* values, std::int64_t rows, std::int64_t samples, std::int64_t first_sample, const selection::ProductForm& cells, const double* weights, const Request&);
// a measurement's ingredients over the members of another selection given in the first one's member numbers (events over a grid, a coarser level)
Ingredients merge(const Ingredients& over_members, const selection::Elements& groups, const Request&);
double value(const Ingredients&, std::int64_t row, std::int64_t member, const std::string& measure);
}
```

| node `reduce` | type |
|---|---|
| in `x` | `field<D; V>` or `measurement<members(G) × R>` |
| in `over` | `selection<S on F ⊆ D>` (any of the five types; any family) |
| in `exclude` (optional) | `selection<spans on T>` (bad segments: members lose samples, never a factor) |
| in `weights` (optional) | `field<S; scalar; area>` |
| out `values` | `measurement<D with F replaced by members(S); measures [...]; ingredients>` |

Move `measure` (type spec §5): the only move that removes a factor. Over a time grid on the streamed factor it accumulates across blocks; elsewhere it
is pointwise in time. Over an irregular selection whose members are given on a grid's members (events detected over G), `x` is the measurement over G
and the node calls `merge`. Parameter `measures` (no default: the caller's or the preset's); `weight` stated (none | barycentric | voronoi).

### 2.3 `roll_up`: through a grid's closed-form hierarchy, or across a group's subjects

```cpp
// compute/include/nxr/measurement/roll_up.h
namespace nxr::measurement {
// the groups of a coarser level as Elements in the finer level's member numbers: dyadic (code >> (D − l)), time (code >> m from the line's first tile),
// one factor of a product (unpack, shift, repack), the whole (level −1: one member)
selection::Elements coarser_groups(const selection::Selection& grid, int from_level, int to_level);
selection::Elements coarser_groups(const selection::Selection& product, std::size_t factor, int from_level, int to_level);
// the same members in several subjects: ingredients summed code for code (a stack of subjects is one more factor, rolled to one member)
Ingredients across(const std::vector<const Ingredients*>& subjects, const Request&);
Ingredients roll_up(const Ingredients& fine, const selection::Elements& groups, const Request&);   // = merge
}
```

| node `roll_up` | type |
|---|---|
| in `x` | `measurement<members(G at level a) × R>` (G a GRID; refused over an irregular selection: `rolling_up()`) |
| out `y` | `measurement<members(G at level b) × R>`, b coarser than a, or b = whole |
| (group scope) in `x` per subject | the same measurement in each subject, over the group tree's codes (`select.dyadic_surface` rule transfer) |

Parameter `to`: the coarser level as a QUANTITY (a tile size, a duration, `whole`, `pieces`), resolved in closed form (section 3.3).

- Replaces: `reduce_cells` (the largest tile per time tile = roll_up of a cell measurement along the space factor to `pieces` or `whole`, merge max),
  `nxr::tower::rollTime`, io's `rollup::space` and `rollup::time` (`io/rollup.h`), group.cpp's subject merge (`pyramid_reduce`).
- Read-time roll-ups (a query asks for a coarser level than stored): io's measurement query returns the stored finest ingredients with the grid's
  closed-form description; the query face (`pipelines`, which meets io and compute) applies `roll_up` and the derived measures. io keeps no copy of the
  arithmetic (decision D1, section 12: an amendment to ARCHITECTURE.md's description of io's query).

### 2.4 `summarize`: members → statistic

```cpp
// compute/include/nxr/measurement/summarize.h
namespace nxr::measurement {
enum class Combine { each, pool, mean };
struct Summaries { std::vector<std::string> names; Combine combine; int ddof; };       // mean, sd, median, mad, percentile:P, min, max, skewness, kurtosis, peak, count
Eigen::MatrixXd summarize(const Eigen::MatrixXd& values /* rows × members */, const Summaries&);   // NaN members skipped and counted
}
```

| node `summarize` | type |
|---|---|
| in `x` | `measurement<members(P) × R>` (any family: across an irregular set it is a `summary` statistic; never of a field) |
| out `y` | `statistic<axes R × summaries; of x; computed from P>` |

Replaces the step `summarize` (registry.cpp) and `summarize_measurements` (verb_steps.cpp), and the across-member functions of `stats.h` callers
(they stay as the summaries' arithmetic).

### 2.5 `estimate`: a statistic from a partition of the stream (Welch: segment means)

```cpp
// compute/include/nxr/measurement/estimate.h
namespace nxr::measurement {
// the mean over a family of segments (may overlap) of a per-segment quantity, folded block by block; segments cut by an excluded span are skipped and counted
struct SegmentMean { void add(std::int64_t segment, const Eigen::Ref<const Eigen::MatrixXcd>& value); Eigen::MatrixXcd mean() const; std::int64_t used, skipped; };
struct Histogram { std::vector<double> edges; void add(const float* v, std::int64_t n); std::vector<std::int64_t> counts; };   // fixed bins (log bins as nxr)
}
```

| node `estimate` | type |
|---|---|
| in `x` | `field<M × T>` (or its per-segment transform: `field<M × members(segments) × frequency>`) |
| in `over` | `selection<spans on T; family>` (`select.time_grid` rule `segments`, or a frame grid) |
| in `exclude` (optional) | bad spans |
| out `y` | `statistic<axes …; method welch | histogram | sample; computed from over>` |

The per-segment quantity is a pointwise transform (taper and FFT; X Xᴴ; |X|²; the mode weights), declared as nodes; `estimate` only means or counts.
`welch::Welch`, `ComplexWelch`, `LinearWelch` and `CrossWelch` become compositions of the transform and `SegmentMean` (one accumulation), and the
histograms (`core_histogram`, `relative_histogram`, `histogram::histogramTiles`) use `Histogram`.

---

## 3. Quantities, charts and units: index versus coordinates versus metric versus units

Ported from the Studio's schema (`space` and `coordinate_system`; the entity `space.ts`): a SPACE has several coordinate maps (charts), one the
REFERENCE, the others defined by a transition map (`identity`, `affine`, `log`, `power`, `operator`), each with a unit. Examples: the frequency line
(reference `linear`, Hz; `log10`); a spectral line (reference `eigenvalue`, 1/m²; `wavelength` = 2π/√λ, a `power` map, in mm); time (`clock`, s);
head space (`scs`, m; `mm` by an affine scale). The METRIC is the geometry's (spacing, or induced from positions); an INDEX (a sample, a vertex, a
level, a member number) is a position in an array, never a coordinate.

### 3.1 The rule

1. Every selection parameter in a definition, preset or command is a QUANTITY: a number (or range) with a unit, in a named chart of the space it
   lies in: `8-12 Hz`, `2-5 s`, `tile 20 mm`, `wavelength 40 mm`, `parts 16/hemisphere`, `area 100-400 mm²`.
2. A bare level index is never a parameter (the wrong form: `--band 8-12 --level 5`, Hz beside an index). An index is allowed only through the
   explicitly named `index` chart (`level index:5`, `member index:3`), for diagnostics and tests.
3. Grid levels are RESOLVED from quantities in closed form. The run records, in provenance, the stated quantity, its chart, the resolved level and
   the SNAPPED quantity (what the level actually is: `2^k / fs` seconds, the level's measured mean tile diameter).
4. A quantity's unit must be one of its chart's (or convertible by the chart's transition); otherwise the run is refused naming the charts of that
   space.

### 3.2 How a quantity is written (one parser: `compute/include/nxr/quantity.h`)

```cpp
namespace nxr::quantity {
struct Quantity { double lo, hi; std::string unit, chart; };        // a value has lo == hi; chart: linear, log10, clock, eigenvalue, wavelength, area, diameter, parts, index
Quantity parse(const std::string& text);                             // "8-12 Hz", "2.5 s", "20 mm", "wavelength 40 mm", "parts 16", "index:5"
double to_reference(const Quantity&, double v, const Chart& chart);   // by the chart's transition (identity, affine, log, power)
}
```

The charts are read from io (the space's `coordinate_system` rows) and handed to compute as data (`Chart{name, unit, transition, params}`).
Spaces lacking a chart get it from the definition that makes the space (e.g. `eigenvalue` and `wavelength` on the Eigenvalue line; `linear` and
`log10` on the Frequency line), as the Studio's schema records them.

### 3.3 Levels resolved from quantities (closed form)

| grid | stated quantity | resolved level | snapped quantity recorded |
|---|---|---|---|
| a line's dyadic time grid | a duration d (s) | k = first with `2^k ≥ d·fs` (`natural_level`) | `2^k / fs` s |
| the same, from a band | a band (Hz, any frequency chart) | k = `time_level_of_band` (4 cycles of the centre) | the tile's seconds and the band |
| frames | a band and `2^s` frames per tile | k − s | the frame's seconds |
| a constant-Q ladder level | a band (Hz) | the ladder level whose band holds the band's centre | the level's `resolves_hz` |
| a surface's dyadic grid | a tile diameter (mm) | the coarsest level whose MEASURED mean tile diameter is at most it (Rheome `tileMM`) | the level's diameter |
| the same | a tile area (mm²) | the level whose mean tile area is nearest below | the level's area |
| the same | a wavelength (mm) or eigenvalue (1/m²) | the Weyl level holding it (`weyl_level`) | the level's Weyl band |
| the same | parts per hemisphere | l = log2(parts) (exact powers only, else refused) | the parts |
| the same | a named scale (hemisphere, lobe, region, gyrus: `scales.h`) | the scale's wavelength, then as above | the scale and its band |
| a roll-up target | any of the above, or `pieces` / `whole` | as above; `whole` = level −1 | as above |

Compute: `grid.h` gains `level_of_duration`, `level_of_diameter`, `level_of_area`, `level_of_wavelength`, `level_of_parts`, each returning
`{level, snapped}`; the existing `natural_level`, `time_level_of_band`, `weyl_level`, `cortical_level_of_extent` stay as their cores.

### 3.4 Events filtered by extent

Irregular members are filtered by their EXTENT in physical units (`duration: 2-5 s`, `area: 100-400 mm²`, `radius: ≥ 5 mm`), and each member is
indexed (on request) on the grid level its extent implies (`address_on_line`, `dyadic_address`, at the level of section 3.3 for its extent). The
filters of `select.events.*` (`min_s`, `max_s`, `min_cells`, `min_area`, `durations`, `merge_s`, `min_cycles`) become quantity filters: `duration`,
`gap` (merge), `area`; a filter in cycles is a duration in the band's chart (`cycles` of the band's centre), named as such.

### 3.5 Mapping of today's index parameters

| parameter today (where) | becomes |
|---|---|
| `--depth` (tiles, pyramid, phasegradient, inverse), `depth` | `finest: <tile diameter mm | area mm² | parts>` |
| `--atlas-depth` (dataset create) | `finest` of the group grid, as a quantity |
| `--lc-levels`, `lc_levels` (group_tree) | the levels of the Levi-Civita measurement as tile diameters (mm) |
| `--level`, `level` (crossspectrum_analytic, lagfree_envelope: tile of 2^level samples) | `tile: <duration s>` or `band: <Hz>` |
| `level` (patches) and `space_level`, `--space-level` (cortical_cells, global_bursts, joint, cell_phase_current, cellquadrature, vortex_*), `space_levels` | `spatial_scale: <tile diameter mm | wavelength mm | eigenvalue 1/m² | named scale>` (one, or a list) |
| `levels`, `reference_level`, `--reference-level` (pointspread, spatial_*) | `spatial_scales` and `reference_scale` as quantities |
| `time_level`, `--time-level` (joint) | the band (Hz) that couples it, or `tile: <s>` |
| `frame_level`, `--frame-level` (frametiling, pyramid, group_tree) | `frames: band <Hz>, 2^s per tile` (or `frame: <s>`) |
| `frames_per_cycle` (frametiling) | `frames_per_tile` as a power of two (decision D3) |
| `apply_kernel`'s `level`, `levels` | the spatial band as a quantity, coupled to its level by `space_level_of_band` |
| `measure query --space-level --time-level --band --spatial-band` | quantities; the level printed with the snapped quantity |
| `min_cells` (global_bursts, cortical_event_pairs) | `area: ≥ <mm²>` |
| `order`, `modes` (chebyshev degree, package size) | stay: they are method parameters, not selections |
| `segment_seconds`, `halo_seconds` | stay (already quantities) |

---

## 4. Field chains, `measure`, and its presets

### 4.1 Field chains

A FIELD CHAIN is a definition with header `kind: chain` that makes ONE kind of pointwise field (no reduction, no detection). It can be `run` alone
(the field is stored, materialized) or COMPOSED into another definition (its steps inlined before the consumer's, nothing stored). Composition uses the
definition language's `chain:` input (section 4.2); the chain's provenance (its name and resolved parameters) is part of the consumer's cache key.

| chain | inputs | parameters (quantities) | output |
|---|---|---|---|
| `band_analytic` | a recording | `bands` (a band set, made by `select.bands`), shape | `field<sensors × members(bands) × T; complex>`: one transform a block shared by all bands, then the bands selected |
| `cortical_band` | a recording, an inverse kernel, (optional) a surface grid, (optional) a frame grid, (optional) canonical frames | `bands`, shape, `spatial_scale` (folded into the kernel once: M = (F ⊗ I)K, D106), transform route (`auto` eigenmodes first, `chebyshev` announced, `exact`), envelope (`temporal` / `spatiotemporal`) | `field<cortex × members(bands) × T (or frames); complex vector3 | scalar>` |
| `member_activation` | a recording, a kernel, a filter bank, a frame grid | the bank member (Hz) | `field<cortex × frames(the frame grid's)>` (optical flow's input) |
| `band_outer` | composes `band_analytic`, then the pointwise outer product z zᴴ (the general A M Aᴴ algebra) | as `band_analytic`, channels | `field<channels × channels × T; complex>` |

The FORM of a field (`magnitude`, `power`, `phase_current`, `outer`, `none`) is one pointwise node `field.form` (the Studio's `fieldForms.ts`,
complex-extended), inserted by `measure` from the measures' registry (section 2.1). `magnitude`, `power`, `phase_current` are its forms (their steps
retire into it).

### 4.2 The one `measure` definition

```yaml
pipeline: measure
version: 2
scope: subject
inputs:
  field:   {field: any, each: true, where: {name: $field}, chain: $chain}        # a stored field by name, OR a field chain composed in ($chain, $chain_parameters)
  over:    {selection: any, where: {name: $over}, make: $make}                   # a stored selection by name, OR made in the run by a maker ($make: {node: select.*, ...})
  weights: {field: any, required: false, where: {name: $weights}}
  bad:     {selection: spans, manifold: field.time, where: {name: "{field}_bad_segments"}, required: false}
parameters:
  field, chain, chain_parameters, over, make, measures (no default), weight (stated), summaries (optional), bad_segments, name
presets: {...}                                                                   # section 4.3
steps:
  chunks: {over: field, do: [field.form per measure → reduce]}
  after:  [roll_up when `to` is given, summarize when `summaries` is given]       # `when:` on a step
outputs:
  values:     {kind: measurement, over: over, of: field}
  statistics: {kind: statistic, of: values, when: $summaries}
```

- Exactly one of `field` / `chain`, and one of `over` / `make`, is given; both are refused with the reason.
- A maker composed in (`make:`) stores its selection get-or-create (a product row, nothing per member); coupled levels are resolved at plan time and
  recorded in the product's factors (section 6.2), never as selection rows.
- With a band set and a coupled product, the run makes one product and one measurement per band member, in one pass.
- A preset is ONLY a named parameter set (the existing `presets:` mechanism). Provenance records the preset's name; the CACHE KEY holds only the
  resolved parameters (today runner.cpp adds the preset to the key: removed), so equal parameter sets are one cached result.
- Several pairs (bands × spatial scales) are several plans of one run (the runner's `each` over a parameter list), each cached on its own.

### 4.3 The presets (named parameter sets of `measure`)

| preset | chain (parameters) | over / make | measures | replaces |
|---|---|---|---|---|
| `cortical_band_cells` | `cortical_band` (band, spatial_scale) | `select.product` {surface grid at the spatial scale's level, time grid `band`} | amplitude, envelope_max | `cortical_cells`, `cortical_cell_pairs` (a list of bands × scales) |
| `cortical_band_cells_current` | `cortical_band` | as above | phase_current, power | `cell_phase_current` (its per-burst values: `reduce` over the bursts, in a second `measure` run over the stored events) |
| `cortical_band_pyramid` | `cortical_band` (band set, frame grid, canonical frames) | `select.product` layout joint {band set, surface grid at `finest`, time grid `frames`} | count, power, phase_current, tensor_* | `pyramid` (array `[band × leaf × frame]` at nxr's atlas node path) |
| `sensor_band_amplitude` | `band_analytic` (band set) | `select.product` {time grid `band` coupled per band, channels elements grid} | amplitude, envelope_max | `band_amplitude`, `measure.band_amplitude` |
| `analytic_cross_spectrum_tiles` | `band_outer` | time grid `band` | sum | `crossspectrum_analytic`, `analytic.tile_cross_spectrum` |
| `channel_span_mean` | none (the stored recording) | `select.product` {the run's span as one member, channels} | mean | `channel_mean`, `filter.channel_mean` |
| `surrogate_lagfree_envelope` | `band_analytic` restricted to the cross-spectrum's channels, form power | `select.product` {time grid `tiles` (duration), the channel set as one member} | power | `lagfree_envelope`, `power_series` |
| `flow_max_amplitude` | `member_activation` | the run's span | envelope_max | `flow_amplitude`, `signal.flow_max_amplitude` (rule 13: stored, then read by name) |
| `psd_channel_mean` | none (a stored PSD statistic) | the channel set as one member | mean | the PSD's channel mean inside the surrogate node |

A plain `measure` with a stored field and a stored grid level by name replaces `measure_over_time`; with a stored map and a parcellation, today's
`measure`.

---

## 5. The group scope

`scope: group` (definition.cpp refuses it today): a definition whose inputs are found in EACH member subject of a group (`per_subject: true`) and
whose outputs are stored in the group's default subject (the template).

- The group is stated by the caller (`--subjects A,B,C`, or a stored group selection by name); no default.
- Every subject's input must lie on the group tree's codes (`select.dyadic_surface` rule `transfer`); the runner checks the codes' grid ids before data.
- Shipped: `roll_up` (group) = `roll_up` node, `across` subjects, merge by measure (replaces `pyramid reduce`, group.cpp `pyramid_reduce`,
  `pyramid_reduce_group`); provenance lists each subject's measurement id and cache key.
- A catalogue dataset (packages): inputs read from each member package, the output written to the group package named by the caller (`--into`).

---

## 6. Storage: one table for values over members, and the migration

### 6.1 The decision

The Studio's schema has ONE place for a value over a selection's member: `selection_measurement` (D40, D48, D68; array-backed D96). It stays the one
table. The `statistic` table (nxr-neuro's) holds STATISTICS only: values on axes that are not a selection's members (a Welch PSD on channels ×
frequency, a covariance, a histogram's bins, the summaries across members). Its `partition_selection_id` keeps one meaning: PROVENANCE, the members it
was computed from (Welch's segments, the selection summarized), never an axis.

- The checker refuses a `statistic` output whose axes include `members(P)` (it is a measurement: declare `kind: measurement`).
- A per-member value that is not a scalar (a window's z zᴴ, a tile's crossing counts) is an array-backed measurement `[rows × members × components]`,
  the components naming the value's parts (`re, im`; matrix entries), the zarr attributes holding `value_type`, the rows' axes and the producer.
- `selection_measurement` gains no producer column: the array node's attributes and the unit's producer record carry provenance (as today); a scalar
  row names its run in `computed_by`.

### 6.2 Schema 53

- `selection_factor` gains `level INTEGER` (null: the factor as a whole): a product's factor may be a LEVEL of a stored grid (the line's time grid,
  the surface's dyadic grid), so coupled levels need no rows of their own. `params_json.product` entries gain `level` and the snapped quantity.
- A line's time grid: one `ladder` row `<line> time grid` with `params_json.grid {kind: time, origin, rate}` (closed form, all levels); the chunk grid
  and the constant-Q ladder keep their rows (the constant-Q ladder with the Studio's level attributes).
- `selection_measurement`: no column change; the `statistic` rows with a member axis move into it (6.3).
- `user_version` 53; `Database::open` chains `migrate_52_to_53` (DDL only); the data step is the written migration below.

### 6.3 The migration (written, not run; the fixture alone is migrated in the final round)

`migrate_measurement_storage(folder, {dry_run})` (`io/include/nxr/io/migrate.h`, `io/src/migrate_measurement.cpp`; command
`package migrate-measurements --dataset F [--dry-run]`), per subject:
1. Each `statistic` row with `partition_selection_id` whose `axes_json` has an axis on that selection's members: write one `selection_measurement`
   row per measure (`selection_id` = the partition, `of_field_id` = the statistic's field, `of_statistic_id` its parent statistic if any, `measure`,
   `array_path` = the array, `component` null); when the member axis is not the array's second axis, the array is rewritten into
   `[rows × members × components]` at `P.migrated`, then replaces P (a rerun finishes an interrupted replacement, as `migrate_selection_storage`).
   The statistic's `options_json`, `producer_json`, `axes_json`, `value_type`, `components_json`, `unit` go into the array's attributes. The statistic
   row is removed; a `statistic` whose parent was it is re-pointed to the measurement (`of_statistic_id` null, `of_field_id` kept, `options.of_measurement`).
2. Products whose factors are level rows made only for a coupling (`params.joint.*` cells, `<name>_<lo>_<hi>_tiles`) are rewritten as products of
   (grid, level); measurements keep their codes (the code arithmetic is unchanged).
3. Frame tilings on the day-anchored tower (`params.atlas.frame_level`) are kept read-only and marked `retired_grid: tower` (their measurements stay
   readable; new runs use the line's time grid).
4. Meta `measurement_storage = "members"`. The store refuses to write a per-member statistic into a dataset without it.

---

## 7. Lines: one Frequency line, one Eigenvalue line (as nxr)

- FREQUENCY: the dataset's `Frequency` line in the default subject (nxr's `convert.frequency`; `nxr/bst/frequency.h`): a line in Hz, origin 0,
  spacing 0.5 Hz, 0-500 Hz, charts `linear` (reference, Hz) and `log10`. Every band set is spans on it in Hz. `select.bands` no longer makes a
  per-recording line (`{recording}_fft_freq`, the one-chunk resolution line, retire). A transform's bins are the made line of its output domain
  (`signal.rfft`: bin k at k·rate/n), a factor of the field's type, stored only as a statistic's axis.
- EIGENVALUE: the dataset's `Eigenvalue` line in the default subject (`io/spatial_bands.h`): 1/m², charts `eigenvalue` (reference) and
  `wavelength` (mm, power map 2π/√λ). Spatial band sets are spans on it, owned by the surface's subject. An operator's eigenvalues are a FIELD of the
  operator (`of_operator_id`) on its mode axis, read against this line when a band resolves modes.

---

## 8. Absorbed items

| rule | what this spec makes of it |
|---|---|
| 22 | nxr's stored layouts: the pyramid preset writes one array `[band × leaf × frame]` at `source-time/<recording>__<kernel>_atlas/<tree>/{power,env,envmax}`; declared outputs gain `path` and `axes` |
| 23 | `band_analytic` and `cortical_band` do ONE transform a block shared by all bands, then select the bands (no FFT per band) |
| 24 | constant-Q dictionaries keep `ConstantQLevelInfo` (low_hz, high_hz, centre, window, …) as entry attributes |
| 25 | lines made where nxr and the Studio make them (section 7) |
| 26 | values on pairs of members: a measurement over `select.product {level, level}` (tiles × tiles), not an `indices` set with hand-made pair numbers (supersedes the `a·tiles + b` pairs selection) |
| 28 | the glued surface is a manifold made by its own definition `glue` (a `make` move: two hemispheres and the bridge), named as an input; `--glued` (eigenmodes) and `glued` (apply_kernel) are removed |
| 30 | the executor's node arrays carry their input's precision (float32 in, float32 arrays; `ng::Array` typed storage), no blanket double |
| 31 | eigenmodes stores the eigenvalues as a field of the operator, besides the mode axis's coordinates |
| 32 | the PSD's channel mean is a stored measurement read by name (`psd_channel_mean`); `lagfree_envelope` absorbed (`surrogate_lagfree_envelope`); `surrogate.*` are native nodes like every other; the fused operator is stored through io (an operator with provenance, found by its key) instead of a static cache; a size entry where the data has no share stays NaN; the type checker carries the "awaits its operator" rule |

---

## 9. The mapping table: every existing piece and its new form

### 9.1 Definitions

| today | new form |
|---|---|
| `tiles` | retired: `select_dyadic_surface` (rule subject) |
| `group_tree` | retired: `select_dyadic_surface` (rule group) |
| `group_tiles` | retired: `select_dyadic_surface` (rule transfer) |
| `joint` | retired: `select.product` composed into `measure` (or `select_product` run alone) |
| `frametiling` | retired: `select_time_grid` (rule frames) |
| `pyramid_bands` | retired: `select_bands` (rule frequency) on the Frequency line |
| `spatial_bands` | retired: `select_bands` (rule levels) on the Eigenvalue line |
| `pyramid` | retired: `measure` preset `cortical_band_pyramid` |
| `cortical_cells`, `cortical_cell_pairs` | retired: preset `cortical_band_cells` (pairs: lists of bands and spatial scales, one plan each) |
| `cell_phase_current` | retired: preset `cortical_band_cells_current`, then `measure` of it over the stored bursts |
| `cortical_event_pairs` | retired: `global_bursts` with lists of bands and spatial scales (one plan each) |
| `measure_over_time` | retired: `measure` (stored field, stored time grid level) |
| `channel_mean` | retired: preset `channel_span_mean` |
| `band_amplitude` | retired: preset `sensor_band_amplitude` |
| `crossspectrum_analytic` | retired: preset `analytic_cross_spectrum_tiles` |
| `lagfree_envelope` | retired: preset `surrogate_lagfree_envelope` |
| `flow_amplitude` | retired: preset `flow_max_amplitude` |
| `measure` | kept, version 2 (section 4.2): the one measuring definition |
| `statistic` | renamed `summarize` (the summarize op on stored measurements) |
| `frames` | renamed `surface_frames` |
| `band_bursts` | kept: `field.form` envelope → `reduce` over time grid `band` → `summarize` → `select.events.spans` → `reduce` over the bursts → `summarize` |
| `global_bursts` | kept: `cortical_band` chain → `reduce` over the coupled product → `roll_up` to `whole`/`pieces` (max: the trace) → `select.events.spans` → `risefall` → `select.events.spans` (keep_where, stages) → `select.events.regions` (extents) → `reduce` over events → `summarize` |
| `patches`, `spatial_events`, `spatial_events_outside` | kept: detection by `select.events.regions`; their measurements by `reduce`; summaries by `summarize` |
| `vortex_cores` | kept: `select.events.cores`; `tracks` by `select.events.tracks` |
| `vortex_core_threshold`, `spatial_threshold` | kept: histograms by `estimate` (method histogram) |
| `spatial_baseline` | kept: its baseline by `estimate` over the frames outside the bursts |
| `run_repeat`, `tile_crossings` users | kept: outputs become measurements over the level's tiles (section 6) |
| `scaleselection` | kept: its levels as spatial scales (quantities) |
| `cellquadrature` | kept: per-burst values by `reduce` over the bursts |
| `psd`, `crossspectrum`, `rotary_psd`, `sourcepsd`, `sourcepsd_bins`, `sourcetensor`, `hodgepower`, `speedspectrum`, `quadrature`, `noisecov` | kept: their segment means through `estimate` (one `SegmentMean`); segments by `select.time_grid` rule `segments` |
| `phasegradient` | kept: its leaves' sums by `reduce` over the surface grid level stated as `finest` |
| `pointspread` | kept: `levels`/`reference_level` as spatial scales |
| `eigenmodes` | kept: takes the glued surface by name; eigenvalues stored as the operator's field |
| `flow_modes`, `pca`, `positions_transition`, `connectome`, `connectome_vertices`, filters (`bandpass`, `bandstop`, `highpass`, `lowpass`, `notch`), `headmodel`, `inverse` | kept unchanged (no reduction or selection of their own beyond the quantity rule) |
| `surrogate_lagfree_vortex`, `surrogate_lagfree_spatial_events`, `surrogate_lagfree_pairs` | kept: rule 32 items |
| new | `select_dyadic_surface`, `select_time_grid`, `select_bands`, `select_product` (the makers run alone and stored), `glue`, `roll_up` (group), field chains `band_analytic`, `cortical_band`, `member_activation`, `band_outer` |

### 9.2 Nodes and steps

| today | new form |
|---|---|
| `tile_grid`, `group_tile_grid` | `select.dyadic_surface` |
| `frame_tiling` | `select.time_grid` (rule frames) |
| `select.bands`, `pyramid.bands`, `pyramid.spatial_bands` | `select.bands` (rules frequency, levels) |
| `joint.cells` | `select.product` |
| `detect_spans` | `select.events.spans` |
| `fit_gate`, `burst_stages` | `select.events.spans` (rules keep_where, stages) |
| `cell_extent`, `patches` | `select.events.regions` |
| `cores`, `tracks` | `select.events.cores`, `select.events.tracks` |
| `measure` (step), `partition_measures`, `window_measures`, `extent_measure`, `measure_events`, `tile_max` (the per-sample tile mean), `pyramid.band_sums`, `pyramid.frame_samples` | `reduce` |
| `reduce_cells`, `tile_max` (the max over tiles) | `roll_up` |
| `summarize` (step), `summarize_measurements` | `summarize` |
| `core_histogram`, `relative_histogram`, `baseline` | `estimate` |
| `welch.*` segment accumulation, `analytic.tile_cross_spectrum` | `estimate` (`SegmentMean`); the latter a preset (reduce of `outer`) |
| `measure.band_amplitude`, `filter.channel_mean`, `power_series`, `signal.flow_max_amplitude` | `reduce` (presets of `measure`) |
| `magnitude`, `power`, `phase_current` | `field.form` |
| `pyramid.band_analytic`, `pyramid.estimate` | `cortical_band` chain (`analytic`, `apply_kernel`, `frames`) and the declared field output |
| `analytic`, `envelope`, `apply_kernel`, `frames` (resample onto a frame grid), `collect`, `relative`, `at_level`, `risefall`, `scale_selection`, `tile_crossings`, `run_repeat`, `cellquadrature`, `pointspread`, `patch_summaries`, `hodge_dirac`, `surface_frames`, and the native `fft.*`, `signal.filter`, `flow.*`, `hodge.*`, `meeg.*`, `signal.pair_series`, `stats.pca`, `timebank.*`, `signal.source_planes`, connectome nodes | kept (`apply_kernel` loses `glued`, its `level`/`levels` become spatial scales; `patch_summaries` and `tile_crossings` emit measurements; `surrogate.*` become native nodes) |

### 9.3 Hand-wired verbs and command-line routes

| today | new form |
|---|---|
| `constantq compute` (cli/nxr_neuro.cpp `constantq_main`, pipelines `constant_q_write`) | retired: `run select_time_grid --rule constant_q` |
| `phasegradient`, `cyclepattern`, `opticalflow` routes (cli/nxr_neuro.cpp, the dispatch at about line 506; `phasegradient_compute`, `cyclepattern_compute`, `opticalflow_compute` in meeg.cpp) | retired: `run phasegradient` (definition); cyclepattern = `estimate` of the window cross-spectrum series (`select.time_grid` rule windows) then `summarize` (method `cycle_pattern`: the eigendecomposition across windows), one statistic with the spatial bands as an axis; opticalflow = `run flow_modes` after `run measure --preset flow_max_amplitude` |
| `pyramid reduce` (`pyramid_main`, group.cpp `pyramid_reduce`, `pyramid_reduce_group`) | retired: `run roll_up` (group scope) |
| `measure query`, `measure export` | kept (reads); parameters as quantities; the hard-wired `timeLevel = 0` goes; read-time roll-up through compute (section 2.3) |
| `surrogate_lagfree_*_compare` | kept (early stopping loops); rule 32 |
| thin wrappers `tiles_compute`, `group_tree_compute`, `group_tiles_compute`, `frame_tiling_compute`, `pyramid_leaves`, `pyramid_leaves_record`, `frame_cells`, `frame_sums_write`, `pyramid_bands_selection`, `pyramid_spatial_bands`, `group_leaves`, `tile_summaries_write`, `stored_map`, `joint_bits` | retired; callers (dataset.cpp, spatial_band.cpp, tests) run the makers' definitions through `run_shipped` or find the stored selection by name |

### 9.4 Compute functions

| today | new form |
|---|---|
| `partition::accumulate`, `meanOverMembers`, `medians`, `measure`, `field_measures`, `known_measure`, `Sums` | `measurement::reduce`, `measurement/measures.h` |
| `partition::reduceTiles`, `crossSpectrum`, `sumOverMembers`, `analyticSums`, `tensorSums` | `measurement::reduce` (Rule / ProductForm forms; the outer product is the `outer` form) |
| `pyramid::bandSums`, `tensorBandSums`, `addSampleCounts`, `frameOfSamples` | `measurement::reduce` over the joint `ProductForm` |
| `pyramid::analyticBand`, `spectrumOf`, `bandGains`, `octaveBands` | kept as the chain's transform (`octaveBands` → `selection/bands.h`) |
| `band_amplitude::bandAmplitude`, `bandAmplitudeOverTiles`, `bandBins` | `reduce`; `bandBins` → resolving a band on a transform's made line (`selection/bands.h`) |
| `ladder::scalarSummary`, `frameSummary`, `vectorSummary`, `tileAreas`, `tileCentres` | `reduce` (members' own `area`, `centre`); `leviCivita`, `lcResultant` → `reduce` measure `levi_civita` over the level × level product |
| `intrinsic::tileFlux`, `tileCirculation` | `reduce` measures `outflow`, `circulation` |
| `integrals::flux`, `circulation` | kept (per-element forms the measures sum) |
| `intervals::rollupEvents`, `extent::rollup` | `measurement::merge` over the events' groups |
| `extent::areaByFrame`, `pathLength`, `windowRegions`, `standardize` | `reduce` members' own measures (`area`, `path_length`); `windowRegions` → `select.events.regions`; `standardize` stays (relative to a stored baseline) |
| `tower::rollTime` | `measurement::roll_up` |
| `tower::frameLevelFor`, `levelForSeconds`, `makeFrameTiling`, `frameSpan`, `parent` | retired: `time_level_of_band`, `level_of_duration`, `time_grid` |
| `frame_grid::frameGrid`, `frameIntervals` | `select.time_grid` (`time_grid`, `resolve_intervals`) |
| `histogram::histogramTiles` | `measurement::Histogram` |
| `welch::Welch`, `ComplexWelch`, `LinearWelch`, `CrossWelch` accumulation | `measurement::SegmentMean` |
| `selection::constant_q_bands`, `eigenvalue_bands`, `scales_by_wavelength`, `scales_by_parts` | `selection/bands.h` (rules frequency, eigenvalue, wavelength, parts) |
| `selection::time_grid`, `dyadic`, `transfer`, `constant_q`, `constant_q_design`, `product`, `joint`, grid.h level functions | kept (the makers' cores); grid.h gains the quantity resolvers (section 3.3) |
| `synthesis::channelMeanSpectrum` | retired (rule 32: a stored measurement read by name) |
| `stats.h` | kept as `summarize`'s arithmetic |

### 9.5 io

| today | new form |
|---|---|
| `io/rollup.h` (`rollup::space`, `rollup::time`, `merge_of`) and the roll-ups inside `selection_measurement_rows.cpp` | retired: the query returns finest ingredients and the grid's closed form; roll-up and derived measures in compute, applied by the pipelines' query face (D1) |
| `statistic` rows with `partition_selection_id` and a member axis | `selection_measurement` array-backed rows (6.3) |
| `selection_factor` | gains `level` (schema 53) |
| `MeasurementOut` (task.h; it declares `band` twice) | one `band`; gains `value_type`, `components`, `of_statistic_id`, `level` |
| `nxr/bst/frequency.h`, `io/spatial_bands.h` | kept: the one Frequency and Eigenvalue lines; `spatial_bands.h`'s joint band key → a product of the two band sets |
| `io/constant_q.h` | kept (the format's own tile arithmetic), joined to the line's time grid row |
| the static operator cache (registry.cpp `operator_cache`) | an operator stored through io with its provenance key |

---

## 10. Tests (synthetic data only; written with the batches, run in the final round)

- compute: `reduce` on every form against hand sums; `merge`/`roll_up` against io's old `rollup` goldens (`gen_rollup_golden.py`); `SegmentMean`
  against scipy Welch goldens (existing); `Histogram` against nxr's log bins; quantity parsing and every resolver of section 3.3 (Studio
  `fitLevelOf`/`supportOf`, Rheome `natural_level`, `jointcell` numbers).
- pipelines: each maker node's dry run (types), the coupled product per band, `measure` with a chain composed and with a stored field giving the
  same numbers, presets out of the cache key (two presets with equal parameters hit one result), group-scope roll-up on two synthetic subjects.
- io: schema 53 round trip; the migration on a synthetic schema-52 dataset (dry run and real) including an array rewrite.
- cli: retired names refused naming their replacement; quantities required where a level index was.

## 11. Checklist (the architecture policy)

- Each new compute function takes matrices and resolved selection forms from anywhere: yes (section 2).
- A pipeline runs the same chain on another input by naming it: yes (`measure` takes any field or chain and any selection).
- Every loop over samples or chunks, and every file access, is inside io: yes (the makers read no data; `reduce` folds blocks the chunk service hands).
- Every choice of data (signal, source, span, selection, scale) is the caller's, stated as a quantity: yes (section 3).

## 12. Decisions for the maintainers (not taken here)

- D1. Read-time roll-ups move out of io (one implementation, in compute; the query face applies it). This amends ARCHITECTURE.md's description of
  io's measurement query. Confirm.
- D2. The day-anchored tower origin of frames retires with `frameLevelFor`: all time grids start at the line's first sample. Confirm (frames of
  different recordings then no longer align on the clock).
- D3. Frames per tile `2^s` in place of frames per cycle (36): the default s for phase work (s = 7 gives 32 frames a cycle at 4 cycles a tile).
- D4. Band sets live in the default subject (on the dataset's lines, as nxr); spatial band sets in the surface's subject on the shared Eigenvalue line.
- D5. Preset names (section 4.3) and the rename of `statistic` to `summarize`.
