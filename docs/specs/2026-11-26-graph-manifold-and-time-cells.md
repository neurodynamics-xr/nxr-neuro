# The fibre network as a graph manifold, and the flows' time cells on a stored time grid (2026-11-26)

**Status:** design written before the code (CLAUDE.md, *Before designing a node or a pipeline*), then built. Two approved gaps of the storage of flow results:
G1 (issue #281, the network as the third space) and G7 (issue #282, the time cells stored). Also #280 (`copy nodes` carries the measure declarations).

**Searched first** (`tools/what_exists.py`): "graph manifold eigenmodes", "graph laplacian weighted", "hodge decomposition graph cycle harmonic", "write manifold graph
edges", "time cells stored measurement", "box window time grid", "vertex area barycentric", "roll up product factor". Found and REUSED: the schema's manifold type
`graph` and topology kind `graph` (`model.sql`), io's manifold writer (`ManifoldSpec`, `TopologySpec` with rank-1 cells: `writer.h`), the cell complex writer
(`task::ComplexOut`, the pattern a graph follows), `nxr::select::bisection::vertexAreas` (the tile areas), `run eigenmodes` and its four-class output form, the runner's
store-on-miss (`made_by`), `select_time_grid --rule tiles`, `select.product` (a time grid and kernel sets are both factors it takes), the compute roll-up of one product
factor (`nxr::measurement::coarser_groups`) and io's (`rollup::product_factor`), the declared measures (`flow_measures.h`). Nothing found writes a graph manifold, solves
a graph's Hodge–Dirac modes with node and edge weights, or stores a flow's time cells.

## G1. The fibre network as a graph manifold

### What is stored (by `tile_network`, the maker)

| class | name | what |
|---|---|---|
| manifold, type `graph` | `<name>` | the network: its vertices are the tiles of the end surface's dyadic grid at the level stated as a quantity (`--scale "parts 32"`, `"area 400 mm2"`, `"20 mm"`), positioned at the tiles' centroids (the surface's coordinate system); its STORED rank-1 cells (`topology_cell` rank 1, `[E x 2]` tail < tip, in pair order) are the tile pairs joined by at least `--min-streamlines` streamlines; topology kind `graph`, rank 1, its connected components counted |
| field on cells 0 | `<name>_area` | each tile's area (the sum of its vertices' barycentric areas, `vertexAreas`; m²): the node weight, ⋆0 |
| field on cells 1 | `<name>_conductance` | each edge's streamline count / median length (streamlines/m): the edge weight, ⋆1 |
| field on cells 1 | `<name>_streamlines`, `<name>_median_length` | the ingredients of the conductance, stored beside it |
| statistic | `<name>_tiles`, `<name>_pairs` | unchanged: every tile, and EVERY pair of tiles (joined or not: the distance-matched control reads the pairs no fibre joins) |

The metric (area on the nodes, conductance on the edges) is stated in the graph's producer (`metric`). The statistic `<name>_eigenmodes` and the node
`meeg.laplacian_eigenmodes` are removed: the network solves no Laplacian of its own.

### The graph Hodge–Dirac modes (`run eigenmodes --graph NAME`)

A graph is a 1-complex: d0 = B (`[E x V]`, -1 at the tail, +1 at the tip), ⋆0 = diag(area), ⋆1 = diag(conductance), no 2-cells. The Hodge–Dirac operator d + δ on it
has two parts, solved once by one dense generalized symmetric eigensolve (the graph is small: tens to hundreds of tiles):
- **exact (gradient) modes**: Bᵀ ⋆1 B ψ = λ ⋆0 ψ, λ > 0 (the constants, one per connected component, out), μ = √λ; each mode's MIXED FORM `[ψ/√2 ; Bψ/(μ√2)]`
  (the surfaces' convention), orthonormal under diag(⋆0, ⋆1);
- **harmonic (cycle) modes**: the cycle space, `h` with Bᵀ ⋆1 h = 0, ⋆1-orthonormal (E - V + components of them), μ = 0, mixed form `[0 ; h]`. On a graph there are no
  coexact modes: the cycle part of an edge flow is harmonic.

Stored in the SAME four-class form as the surface solve (stem `hodgeDirac__<graph>`): the eigenvalue line `<stem>` (points, coordinate μ, ascending: the cycle modes at
μ = 0 first), `<stem>_eigenvalues` (λ = μ²), `<stem>_kinds` (a labelling: exact 0, harmonic 1), ONE operator `<stem>_eigenvectors` from the line (cell 0) to the graph
(cell key `0+1`): `[(V + E) x K]`, rows 0-forms [0, V) · 1-forms [V, V + E). Store on miss only: `connectome_flow` names the modes with `made_by: eigenmodes`, so the
runner solves them once, as its own run, when they are not stored; a stored solve is read, never solved again.

Graph: `graph (vertices, rank-1 cells) + area + conductance --[meeg.graph_hodge_dirac]--> line, eigenvalues, kinds, eigenvectors`. `eigenmodes.yaml` takes one of
`--surface` (the surfaces' step, unchanged) or `--graph` (`one_of`).

### What reads it

`connectome_flow` names the network (`--network NAME`: the graph and its pairs) and reads the stored graph, its conductance and its modes (made first when missing).
`meeg.network_flow` no longer reads a Laplacian's modes: per atom, the mean edge flow F on the graph's edges gives
- the divergence Bᵀ F per tile (as before);
- the edge flow's coefficient on every mode, its projection on the mode's 1-form part φ_k: c_k = ⟨F, φ_k⟩⋆1 / ⟨φ_k, φ_k⟩⋆1 (2 ⟨F, φ_k⟩⋆1 on an exact mode, as the
  surfaces' `projectOneForm`; ⟨F, h_k⟩⋆1 on a cycle mode), so F = Σ c_k φ_k: the measure `flow_coefficient` (the surfaces' name), rows the graph's eigen line;
- the graph potential φ = Σ_exact c_k (ψ_k/√2) / μ_k (the surfaces' `exactPotential`: the weighted least-squares potential, Bᵀ⋆1 B φ = Bᵀ ⋆1 F, ⋆0-orthogonal to the constants);
- the ⋆1 energies of the gradient part (`weighted_gradient_squares`), the cycle part (`weighted_cycle_squares`, new, declared `sum`) and the flow (`weighted_flow_squares`).

The weights change from the reference's streamline counts to the stated conductance (count / median length); every flow is the same measurement as the surfaces'.

## G7. The flows' time cells on a stored time grid

### Rule (the maintainers, 2026-11-26)

Every windowed quantity uses the ATOM'S OWN WIDTH on the one dyadic time tiling: a window is an atom with a rectangular (box) kernel. There is no separate fixed segment.

### What changes

- `sensor_flow`, `source_flow` and `connectome_flow` take `--time-grid NAME`: a STORED time grid of the recording (`select_time_grid --rule tiles --duration D`, a level
  of the dyadic tiling). They no longer make cells inside the run: `connectome_flow`'s `cells`, `segment_seconds` go; `cell_width` stays as the atom's own window width
  (in the atom's time standard deviations, snapped to the dyadic level) and must not be finer than the grid (refused, naming the level needed).
- The per-tile SUMS are computed on the grid's tiles (each sample once; a tile cut by a chunk's edge completed from the next chunk; the recording's last tile as long as
  it is) and stored as MEASUREMENTS over the PRODUCT of the grid with the members (`select_product --time-grid G --atoms A [--scales B]`; the grid the first factor,
  fastest): sensor and source over grid × atoms × bands, connectome over grid × atoms. Sums are declared `sum` (`phase_current`, `vertex_phase_current_x/y/z`,
  `power_sum`, `cross_spectrum_real/imaginary`), so a coarser level is a roll-up at read along the grid factor.
- The sign-flip test (R1) reads those stored cell sums: per member it rolls the grid factor up to the atom's own level (the sums of 2^(k_a - k) consecutive tiles; a
  cell that would hold the recording's last part tile is left out, as before) and tests the cells. p values are declared `none`.

### Tests

- `test_graph_hodge_dirac` (compute): orthonormality under the stars, a gradient flow's coefficients all exact, a circulation's all harmonic, the potential against a
  direct weighted least-squares solve.
- `test_connectome_flow` (the nodes): the graph, its metric, its modes; network flow from the stored modes.
- `test_graph_network` (integration, a synthetic subject): `tile_network` stores the graph manifold and its fields; `connectome_flow`-style reading through
  `eigenmodes --graph` (store on miss: the second run solves nothing).
- G7: the stored cells of a run roll up to the whole-run sums (every tile summed = the cells of a grid of one tile); the sign-flip test on the stored cells.

## What it cannot resolve

The graph's metric is a choice (area, conductance), stated in the producer; a tile's area depends on the grid's level; the cycle space of a dense network is large.
The time cells are as fine as the grid asked for: a finer grid stores more (memory: rows × tiles × members × measures, float32).
