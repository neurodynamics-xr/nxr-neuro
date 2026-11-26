# Levels of one manifold: addressing and navigation at every level (2026-11-26)

**Status:** design written before the code (CLAUDE.md, *Before designing a node or a pipeline*). First step: the SENSOR MESH × TIME product. No inverse, no
source mapping and no cortex in this step; the code is the same for any triangle mesh, so the cortex needs no new function, only its own checks (section 9).

**Amends** [2026-11-20-time-as-a-mesh.md](2026-11-20-time-as-a-mesh.md) section 2 (the row `time_tiles`, `levels`: "not manifolds: a level is a partition, held by
the selection system"), [2026-11-17-consolidation.md](2026-11-17-consolidation.md) section 6.2 (coupled levels "are never stored as selections of their own") and the
comment of `pipelines/src/dataset.cpp` (a level is no selection of its own). Section 1 states the amendment. It brings those documents in line with
`docs/vocabulary.md`, which already says a level is "a resolution of a manifold, itself a manifold", with its reduction and expansion operators.

**Searched first** (`tools/what_exists.py`): "adjacency", "coarser", "prolong", "galerkin", "quotient", "halfedge", "product code", "roll up", "dyadic", "coarse
complex quotient partition", "tile adjacency halfedge twin next", "parent children dyadic code". Found and REUSED: the dyadic grid and its level dictionary
(`select_dyadic_surface`, `nxr::select::subjectTree`, `selection::dyadic_grid`, `selection::level`), `nxr::select::tileAdjacency` (the stored `lc/L<l>/adjacency`,
which the level's 1-skeleton reproduces on the input mesh), the time grid's closed form (`select_time_grid`, `selection::Grid::Kind::time`), the product's mixed radix
(`selection::product_code`, `product_codes`), the one roll-up (`nxr::measurement::roll_values`, `coarser_groups`), the operator mesh's DEC
(`nxr::operators::assembleDECOperators`, `geometry::IntrinsicSurface::complex`), the graph Hodge-Dirac solve (`nxr::meeg::graphHodgeDirac`, which takes a level's d0,
⋆0 and ⋆1 as they are), `spaceTimePairs` (`select/extent.h`: the product's 1-skeleton). Nothing found builds the quotient of a complex by a vertex partition, its
Galerkin metric, the crossing map of its edges, junction faces, halfedges across tile boundaries, an expansion operator, or a joint address.

## 1. The rule, and the amendment

- **One stored manifold.** The manifold of record is what the data gives, at its finest resolution: the sensor mesh's vertices (the sensors), the recording's samples.
  It is the only manifold row. A level is never stored as a manifold of its own and never gets a name of its own (no `<surface>_level_3`).
- **Each level is a manifold for navigation and computation.** At every level the same questions have answers: a cell's neighbours, its boundary, its area, the
  operators d0 (and d1 where it has faces), ⋆0, ⋆1, the Laplacian, and, where the level makes a valid surface complex, halfedges with twin and next. Navigation is
  the same at every level.
- **A level is a level of a stored grid.** Its identity is the grid's (the grid's level dictionary, as stored today); its cells keep the grid's codes.
- **The amendment.** "A level is a partition, not a manifold" becomes: *a level is a partition held by the selection system (its source of truth, unchanged) AND,
  computed from it, a cell complex in its own right, addressed by (manifold, level, code), never a manifold row.* Coupled levels still need no selection of their own.

## 2. The ladders

| manifold | finest | then | coarser |
|---|---|---|---|
| a surface (the sensor mesh) | its vertices (level `vertices`) | the same vertices with their areas: ⋆0, the dual cell (no new level: the finest level's metric) | the dyadic tiles, levels `depth` .. 0 (2^l tiles a piece; 0 is the coarsest) |
| a time line | its samples, each with its interval to the next: the level-0 tile [i, i+1) of duration dt | | the dyadic tiles of 2^k samples, levels 1, 2, ... (coarser as k grows; the chunk is 15) |

Each ladder keeps its stored label: a surface level is l (2^l tiles a piece, the dyadic dictionary's code), a time level is k (tiles of 2^k samples). The two run in
opposite directions, as they do today; nothing is renumbered. A surface's own vertices are the level `vertices` (`kVertexLevel`, below every dyadic level); a time
line's samples are its level 0. **⋆0 of a sample is dt** (the tile view: a tile's duration is its samples times dt, the last tile as long as the line), not DEC's
half interval at the two ends, so that a level's ⋆0 is exactly the sum of its samples' and a stored roll-up weights each sample once.

## 3. Addressing

- **A cell:** (manifold, level, code). The manifold is the stored manifold row; the level is (the grid, its label) (a dyadic grid's level dictionary entry, a line's
  time grid and k); the code is the grid's member number at that level (a tile code h 2^l + t; a time tile m). The vertex level's code is the vertex index.
- **Parent and child** are the grids' closed forms (`selection.h`): dyadic, parent of (l, c) is (l - 1, c >> 1), children (l + 1, 2c and 2c + 1); the children of the
  finest dyadic level are the vertices whose code is c, and a vertex's parent is its code at the finest level; level 0 has no parent (the whole surface is the
  roll-up to `kWhole`, as today). Time: parent of (k, m) is (k + 1, m >> 1), children (k - 1, 2m and 2m + 1, those inside the line); level 0 has no children.
- **A joint cell** of the product (space at level l) × (time at level k): ((sensor mesh, l, code), (time, k, code)); its packed code is
  `product_code({space cells, time cells}, {space code, time code})`, the selection's mixed radix with the FIRST factor (space) fastest, so a [space x time]
  column-major array (Eigen's layout, `extent.h`'s node = frame x tiles + tile) is indexed by it.
- **Nothing is stored per cell.** Parent, children, a code's cell and a joint code are arithmetic; a cell's members at another level come from the grid's codes.

## 4. How levels are recorded: no table and no column changes

- The grid and its level dictionary stay exactly as `select_dyadic_surface` and `select_time_grid` store them: a dyadic level keeps `w` (its tiles' areas, which IS
  the level's ⋆0 when the fine ⋆0 is the barycentric vertex area), `lc/L<l>/centre` and `lc/L<l>/adjacency` (the input mesh's tile pairs, which the level's
  1-skeleton reproduces exactly on the input mesh).
- The per-level navigation is COMPUTED from the stored manifold and the grid's codes when first needed, and STORED as OPERATORS keyed by (manifold, grid, level):
  `operator.from_manifold_id` = the surface, `from_selection_id` = the grid, the level and its mesh and metric in `params_json` (`{level, mesh, metric}`), named
  `<grid> L<l> d0`, `<grid> L<l> star1`, `<grid> L<l> crossing` and, when the level has faces, `<grid> L<l> d1`. These are the operator table's existing columns; no
  manifold row is written. The maker is the definition `level_complex` (section 7); a consumer names it with `made_by: level_complex` (store on miss).
- A time level needs nothing stored: every operator of a time level is closed form in (samples, rate, k).

## 5. The per-level navigation (compute: `nxr/select/levels.h`, `nxr::select`)

Given a fine complex (vertices with ⋆0, oriented edges d0 with ⋆1, faces in orientation order) and a vertex partition (each vertex's cell at the level, -1 for
none), the level is the QUOTIENT complex, with the Galerkin metric:

| part | what | consistent with the fine mesh because |
|---|---|---|
| cells (0-cells) | the grid's members at the level (empty members stay, with ⋆0 = 0) | a cell's ⋆0 is the sum of its vertices' (Pᵀ ⋆0 P = ⋆0_l, P the piecewise-constant expansion) |
| edges (1-cells) | the pairs (a < b) of cells joined by at least one fine edge, oriented tail < tip, in pair order (the stored adjacency's order) | d0_fine P = E d0_l exactly (a cochain map), E the CROSSING map |
| crossing E | [fine edges x level edges]: +1 for a fine edge from a to b, -1 from b to a, 0 for an edge inside a cell | the boundary between two cells is E's column |
| ⋆1 (the default metric `galerkin`) | the sum of the fine ⋆1 (cotan weights) of the fine edges crossing the boundary | d0_lᵀ ⋆1_l d0_l = Pᵀ (d0ᵀ ⋆1 d0) P: the level's Laplacian is the projected fine one, with the sparsity of the adjacency |
| faces (2-cells) | the JUNCTIONS: each fine face whose three corners lie in three different cells is a level face (a, b, c), oriented as the fine face | d1_fine E = J d1_l exactly, J the map from level faces to their fine face: a face inside a cell or across two cells has zero circulation |
| halfedges | per level face, a -> b, b -> c, c -> a (3f + i); `next` within the face; `twin` the opposite halfedge of the other face on the edge | only where the level is a valid surface complex (below) |

**Valid complex, checked or refused.** Twin and next exist only when the level passes every check: every edge bounds one face (a boundary edge) or two, with
opposite orientations; the faces round every non-empty cell form one fan (a disc round an interior cell, one chain round a boundary cell); and V - E + F over the
non-empty cells equals the fine mesh's Euler characteristic. Otherwise the halfedges are REFUSED with the reason (the edge that bounds no face or more than two, the
cell whose faces form two fans, the Euler characteristic found), and the level stays a 1-complex (cells, edges, d0, ⋆0, ⋆1, d1 of its junctions) on which
neighbours, boundaries and operators still hold. Tiles are one piece (`onePiece`) but not always discs, and on an open mesh (the sensor helmet) a coarse level
often has a boundary arc that meets no junction: these levels are graphs.

**The fine mesh** is a choice of the caller (the method rule): `operator` (default; the intrinsic Delaunay triangulation, whose cotan ⋆1 is non-negative inside,
and whose Laplacian is every fine operator's, the Hodge-Dirac modes included) or `input` (the stored triangulation, whose 1-skeleton is the stored
`lc/L<l>/adjacency` bit for bit). ⋆1 is the raw cotan weight (never floored here; negative fine weights on an open rim are counted in the record
`negativeConductances`). ⋆0 is the caller's: the barycentric vertex area by default in the definition, so the level's ⋆0 is the stored `w`.

**Named alternative metric, not built in this step:** `two_point` (boundary length over centre distance). On a time line Galerkin gives ⋆1 = 1/dt at every level
(one fine edge crosses a tile boundary), the energy of the piecewise-constant expansion; `two_point` would give 1/(2^k dt).

**Cache keys.** A stored level operator's key is its producer record: the surface's and the grid's rows, the level, the mesh, the metric. The time level is closed
form and keyed by (samples, rate, k).

**The time line** (`lineLevel(samples, rate, k)`): cells m = ceil(samples / 2^k), ⋆0 the tiles' durations, edges (m, m + 1), d0, ⋆1 = 1/dt, crossing the fine edge
(2^k (m + 1) - 1, 2^k (m + 1)), no faces, halfedges m -> m + 1 and their twins (no next on a line). It equals `levelComplex` of the line's samples with the
partition i >> k (tested).

## 6. Reduction and expansion per measure (the vocabulary's words: reduction = restriction, expansion = prolongation)

Between two levels of one manifold (`reduction(from, to, rule)`, `expansion(from, to, rule)`; from the vertices or any finer level of the same grid), the rule is the
measure's MERGE RULE as the registry and the declarations hold it:

| merge rule | the quantity | reduction (fine -> coarse) | expansion (coarse -> fine) |
|---|---|---|---|
| `mean` | intensive: a 0-form, a density | ⋆0_c⁻¹ Pᵀ ⋆0_f (the area- or duration-weighted mean) | P: each fine cell takes its cell's value (piecewise constant) |
| `sum` | extensive: an integral over the cell (a count, a power sum, an area) | Pᵀ | ⋆0_f P ⋆0_c⁻¹: the coarse value split by area (or duration) share |
| `max`, `min` | an order statistic | `roll_values` over the level's groups (no matrix) | refused: an order statistic has no expansion |
| `none`, `derived` | p values, ratios | refused (as the roll-up refuses them) | refused |
| 1-cells: `sum` of a flow per edge (a 1-chain, a flux) | | Eᵀ: the signed sum of the crossing fine edges' values; the boundary commutes (Pᵀ d0ᵀ F = d0_lᵀ Eᵀ F) | |
| 1-cells: a difference per edge (a 1-cochain) | | | E: each crossing fine edge takes its level edge's value (d0_f P = E d0_l) |

The 0-cell reductions equal `nxr::measurement::roll_values` with the same rule (the one implementation of a roll-up; `mean` with the weights ⋆0): tested.

## 7. The definition and the node (io: store on miss)

`run level_complex --surface S --tiles G --scale "parts 16" [--mesh operator|input]` (`pipelines/definitions/level_complex.yaml`, the node `select.level_complex`):

    surface (vertices, faces) + tiles (the grid's codes and level dictionary) --[select.level_complex]--> d0, star1, crossing, d1 (sparse operators)

The scale is a quantity (`parts 16`, `"20 mm"`, `"area 400 mm2"`), resolved on the grid's level dictionary (`level_of_quantity`); the run records the stated
quantity, the level and the snapped quantity. Outputs: four sparse operators (section 4; d1 empty when the level has no junction), their producer stating the
level, the mesh, the metric, the counts (cells, edges, faces, negative conductances) and whether the level makes a valid complex and why not. The rerun is a cache
hit. The time line stores nothing.

## 8. The product, and the slot for the dispersion relation

`productLevel(space, time, coupling)`: the product of a surface level and a time level, chosen independently (SEPARABLE):
- cells: joint codes (section 3); `jointCode`, `jointAddress` round-trip;
- neighbours: the space neighbours in the same time tile and the same space cell in the next and previous time tile (`spaceTimePairs`'s graph);
- operators: d0 = [d0_s ⊗ I_t ; I_s ⊗ d0_t] (edges space-edge x time-cell first, then space-cell x time-edge), ⋆0 = ⋆0_s ⊗ ⋆0_t, ⋆1 = [⋆1_s ⊗ ⋆0_t ; ⋆0_s ⊗ ⋆1_t],
  so the Laplacian is L_s ⊗ ⋆0_t + ⋆0_s ⊗ L_t (the weak form of L_space ⊗ I + I ⊗ L_time, `time-as-a-mesh.md` section 1). With P = P_s ⊗ P_t it is again the
  Galerkin projection of the fine product's Laplacian (tested).
- **The coupling slot (typed, not built):** `Coupling { kind: separable | dispersion; speed (m/s); sense (+1 | -1) }`. `dispersion` (a non-separable product whose
  space level is tied to the time level through a stored relation, D_l / Δt_k = c or 2πf_k = c √λ_l) is REFUSED in this step with the reason; the slot keeps the
  type so a later step adds the relation without changing the product's face. A measured dispersion (`sensor_flow_across_frequencies`) stays a statistic.

## 9. The faces

C++ (`nxr::select`, `nxr/select/levels.h`): `levelComplex(fine, cellOf, cells, ladder)`, `lineLevel(samples, rate, k)`, `vertexLevel(fine)`,
`dyadicLevel(fine, codes, depth, slots, l)`; on a level: `neighbours`, `outgoing`, `twin`, `next`, `parent`, `children`, `area`, `laplacian`; between levels:
`reduction`, `expansion`; the product: `productLevel`, `jointCode`, `jointAddress`, `productNeighbours`, `productOperators`.

Python (`nxr_neuro.compute.select`): `dyadicLevel(d0, star0, star1, faces, codes, depth, slots, level)` and `vertexLevel(...)`, `lineLevel(samples, rate, k)`,
`productLevel(space, time, coupling=None)`, each returning a navigable view `Level` (`ProductLevel`) with `cells`, `edges`, `d0`, `star0`, `star1`, `crossing`,
`faces`, `d1`, `junctions`, `valid`, `why`, `neighbours(c)`, `outgoing(c)`, `twin(h)`, `next(h)`, `parent(c)`, `children(c)`, `area(c)`, `star(rank)`,
`laplacian()`, `cell(code)` -> (level, code), and `reduction(to, rule)`, `expansion(to, rule)`; `subjectTree(vertices, faces, depth)` gives the dyadic codes of a
mesh as the grid maker makes them. The fine complex comes from the existing faces (`geometry.surfaceArrays`, `geometry.complex`,
`operators.assembleDECOperators`): the caller composes them; the binding only converts. A stored subject's level is read back as its operators
(`nn.read.operator`) after `run level_complex`.

Left for later steps (each its own work): the cortex (the glued surface's grid; tiles that are not discs at fine levels; the bridge), the `two_point` metric,
the coarse connection (one angle per level edge from `leviCivita`, its holonomy round a junction), measurements read as fields on (manifold, level) through a
members-level link, the dispersion relation, the Studio's and nxr's tile adjacency read from the stored level.

## 10. Tests

- `test_levels` (compute, unit; synthetic sensor helmet: a spherical cap of points meshed by `pointSetMesh`, and a closed icosphere):
  - d0_f P = E d0_l and Pᵀ d0_fᵀ F = d0_lᵀ Eᵀ F for a random F (the boundary commutes on the edges crossing cell boundaries);
  - Pᵀ L_f P = L_l (the projected Laplacian) and Pᵀ ⋆0_f P = ⋆0_l;
  - d1_f E = J d1_l, and d1_l d0_l = 0;
  - with the input mesh, the edges are `tileAdjacency`'s pairs;
  - halfedges on the icosphere's levels: twin(twin(h)) = h, next³ = id, V - E + F = 2; on the helmet, a valid level has χ = 1, and a level whose edge bounds no
    face is refused with that reason;
  - parent and children against the codes;
  - the reductions `sum`, `mean`, `max`, `min` equal `roll_values` with the same rule; expansion `mean` is P, expansion `sum` keeps the total; `max` expansion and `none`
    reduction refused;
  - the time line: `lineLevel` equals `levelComplex` of the samples with i >> k; parent/children; a last partial tile; the reductions equal `roll_values`;
  - the product: joint codes round-trip; neighbours equal `spaceTimePairs`'s; the product Laplacian is the Galerkin projection of the fine product's; `dispersion`
    refused.
- `test_level_complex` (pipelines, integration): `run level_complex` on a synthetic subject's sensor surface and its grid stores the operators with the level in
  their record, a rerun is a cache hit, and the stored d0 equals compute's.
- `bindings/python/tests/test_levels.py`: the Python face on a synthetic helmet and a line: the views, navigation, the identities above in NumPy, the refusals.

## Checklist (the architecture policy)

- Every new compute function takes arrays (a fine complex, a partition) and can be called with matrices made anywhere; nothing reads a file.
- The node takes arrays bound by the runner; storage is io's, through the declared operator outputs; no manifold row is written.
- No loop over samples or chunks: a time level is closed form; nothing streams a recording.
- Every choice (mesh, metric, scale) is the caller's, recorded in the provenance and the cache key.
