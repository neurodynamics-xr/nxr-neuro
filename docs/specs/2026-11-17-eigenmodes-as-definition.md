# The Hodge-Dirac eigenmodes as a definition: `eigenmodes` (spec, 2026-11-17)

Status: **written** (not yet built; compile errors are fixed in the compile round). It replaces the hand-wired `eigen_solve` and the command `eigen solve`. The maths is unchanged (nxr-compute's
`hodgeLadders`, the Studio's glue and intrinsic Delaunay); what changes is who stores: the step describes, the definition places, io writes.

## The graph (written first; the Studio's chain: `glue.ts` glueMesh -> `intrinsicDelaunay.ts` -> the Laplace-Beltrami and 2-form ladders -> the package)

```
surface ──[hodge_dirac, setup]──▶ glued, bridge        (only with `glued`: a cortex of two pieces glued by anatomy)
                              ├─▶ modes                 the mode axis: points, coordinate mu = sqrt(lambda), ascending
                              ├─▶ kind                  field on the axis: 0 exact, 2 coexact               (of_operator: eigenvectors)
                              ├─▶ support               the surface's vertices the solve covers
                              ├─▶ eigenvectors          operator axis -> solved surface [V x K]: the 0-form parts
                              ├─▶ forms2                operator axis -> the complex's faces (cell 2): the 2-form parts
                              └─▶ complex, d0, d1, length, star0/1/2, angle_tail/tip/sum, reference
                                                        the intrinsic cell complex, its sparse operators and the fields on its cells
```

| piece | where | rule |
|---|---|---|
| medial wall, surface | the runner (`Bound::surface`, `Bound::medial_wall`) | the step reads no row and no file |
| glue, Delaunay, ladders, the potentials | `hodge_dirac` calls `nxr::glue::glueMesh` and `solve_hodge_dirac` (nxr-compute's `hodgeLadders`) | arrays in, arrays out; no loop over data |
| what is stored | io's `Unit` through the declared outputs (`eigenmodes.yaml`) | `points`, `complex`, sparse `operator`, `field` with a `cell` |
| the cache | the runner's key: definition, parameters, inputs, and what the step RESOLVES (`resolve`: the surface's mesh key and positions, the medial wall's hash) | a rerun with all the same is a hit |

Eigenmode first (the maintainers' rule) is unchanged: a spatial band whose eigenmodes are not stored has the runner make them first, as this run, with a message; Chebyshev stays the announced fallback.

## The new io kinds (what the stored schema already had, now writable by a run)

| kind | io | schema |
|---|---|---|
| a manifold of `points` with one coordinate each | `PointsOut`, `Unit::points` | manifold type `points`, topology `none`, positions at its path, coordinate system `eigenvalue:<stem>`; replaced with `--replace` |
| a cell complex | `ComplexOut`, `Unit::cell_complex` | manifold type `surface`, topology `simplicial` with the stored cells of rank 1 (edges [nE x 2]) and 2 (faces [nF x 3]) (D92); two edges may join one pair of vertices, so the incidence is read from d0 and d1 |
| a sparse operator | `OperatorOut::sparse`, `from_cell`, `to_cell` | operator `layout` csr (D160), `nnz`, the cell keys of its two ends (d0: '0' to '1') |
| a field on cells | `FieldOut::cell`, integer `Values` | `field.cell` '1' edges, '2' faces; `value_type` integer |

`Unit::replace` of a manifold moves what lies on it aside first (its fields, operators, selections, each with its arrays), so a replaced axis never leaves old arrays where the new ones are written.
Runner features, all generic: link alternatives (`a|b`), `optional` outputs (a node that makes none hands an array marked `absent`), a field on a manifold the run makes (its place is found by the manifold's name),
`facts.names` for the names a step's plan fixes (`{stem}`, `{solved}`).

## Eigenvalues

The schema's eigenvalues are "an operator's by-product on the operator's mode axis" (`field.of_operator_id`). The stored layout of `eigen solve` keeps them as the axis's own coordinates (the points manifold),
which every reader (spatial bands, kernels in the full forms) reads; the kind field is `of_operator` the eigenvectors. A separate eigenvalue field is not stored (see the decisions below).

## Decisions for the PI

1. **`glued` is a parameter, not found from the data.** The step's outputs are declared statically (names, links), and the number of pieces is a fact of the data, so `--glued` says the surface is a cortex of
   two hemispheres; a mismatch is refused with the reason. `eigen_solve` (the library's caller) sets it from the pieces. The alternative, `glue` as its own definition run first, would give the glued surface as an
   input found by name (the rule that inputs are found, not made inside the run), at the price of one more verb.
2. **The complex's name** is `<surface>_glued_intrinsic` when glued and `<surface>_intrinsic` otherwise, as before; its operators and fields are named from it and stored at io's default paths (the operators at
   the root by name, the fields under their manifold), not under the complex's folder as before; readers find them by name.
3. **Memory.** A node's array is in double, so the eigenvectors ([V x K]) and the 2-form parts ([nF x K]) are held in double (twice their stored float32) beside the solve's own. For a 160 000-vertex cortex with
   3000 modes that is about 11 GB for the two; if that is too much, a float32 `Array` (or a streamed readback) is the change, in the executor, not here.
4. **Stored eigenvalues as a field of the operator.** The schema offers `field.of_operator_id` (kind `eigenvalues`) on the mode axis; today the values are the axis's coordinates. Adding the field duplicates them;
   say if the Studio reads one rather than the other.
