# The intrinsic forms as graph nodes (design, 2026-11-19)

Status: **implemented (2026-11-19), option B of section 4**: the runner binds the package's 2-form parts and its complex's options into the eigenmode constant it already
binds (preparation; a node asks for it with `NodeSpec::reads_package_forms`). Built: the `discretisation` option of the `phase_current` form (3.1) and `signal.phase_split` (3.2).

## 1. What is asked

The maths on the intrinsic complex (`compute/include/nxr/signal/intrinsic_forms.h`) exists as pure functions: `phaseGradient(complex, x)` and `phaseCurrent(complex, z)` (1-forms on the
intrinsic edges), `projectOneForm` / `synthesizeOneForm`, `exactPotential`, `coexactStream`, `splitPhase` (pointwise parts only) and the pointwise vertex forms `vertexOutflow`,
`vertexCirculation`, `vertexEnergy`. A pipeline should reach them as registered nodes, so that a definition can compute them on a streamed field and reduce them with
`measurement.reduce` over any partition.

## 2. Constraints from the architecture

- The typed graph's `ManifoldKind` is `points, line, surface, volume, curve, graph, members`: there is **no edge or face kind**. A stored cell complex (the `complex` manifold `eigen solve`
  writes) is typed today as `points` (`kind_of_manifold`, `pipelines/src/step_nodes.cpp`), which says nothing true about edges or faces.
- Existing practice moves edge and face quantities to the vertices before any reduction (`vertexOutflow`, `vertexCirculation`; the measures `outflow`, `circulation`).
- Reductions are measurements over partitions; ratios of reductions are derived at read (the shares of the phase split are now `exact_share`, `coexact_share`, `residual_share`,
  `exact_misfit_share`, derived from the sums `one_form_energy`, `exact_energy`, `coexact_energy`, `exact_misfit_energy`).
- One name per quantity; the discretisation is the operator the caller passes; a node's options default to the reference.

Hence: **1-forms and 2-forms stay inside nodes. Ports carry vertex fields (scalar or vector3) and, where a node gives them, mode coefficients on a line with role `modes`.**
Bound constants (the intrinsic complex, the modes) are built once in `prepare`, as `phase_state` builds the vertex gradient (`pipelines/src/registry.cpp`).

## 3. The graph

```
x  complex field on a surface, streamed [V (x3) x samples]
│
├── signal.field_form {form: phase_current, discretisation: vertex_gradient | intrinsic_edges}            → y  vector3 vertex field (the phase current j)
│     constants (prepare, once): vertex_gradient: the vertex gradient and normals of the surface (today's phase_state)
│                                intrinsic_edges: the intrinsic complex of the same surface (intrinsicComplex(surfaceArrays(positions, faces)))
│     per block: vertex_gradient: phaseCurrent(Z, components, G, N)                                    (the reference, the default)
│                intrinsic_edges: sharp(c, Σ_components phaseCurrent(c, z_component))                  (the edge 1-form, back at the vertices)
│
└── signal.phase_split {one_form: phase_gradient | phase_current, modes: the package}                            → potential   scalar vertex field (rad)
      constants (prepare, once): the package's intrinsic complex and its Modes (P, X2, mu, kind)          → stream      scalar vertex field (rad)
      per block, per column: w = phaseGradient(c, x) [one_form: phase_gradient] | phaseCurrent(c, x) [one_form: phase_current] (no default: the definition states it); s = splitPhase(c, modes, w);                        → propagation vector3 vertex field (rad per length unit)
                             vertexEnergy(c, [w, s.exact, s.coexact, w − s.exact])                       → one_form_energy, exact_energy, coexact_energy, exact_misfit_energy
                                                                                                             scalar vertex fields (rad²)
      (optional, later) coefficients: s.coef on a made line with role `modes`                             → coefficients [K x samples]

each energy ── measurement.reduce {measures: [<its measure>]} over a partition ──> stored sums, OF x
                                     exact_share, coexact_share, residual_share, exact_misfit_share: derived at read from those sums
```

Placement in the four parts:
- loading: x through the chunk service; the package and the surface read by name (io);
- preparation: type checks (x complex, scalar or vector3, on a surface; the modes' triangulation is the complex's: `projectOneForm` refuses another);
  the constants bound once;
- the graph: both nodes pointwise in time (per sample) and pointwise in space (per vertex outputs); no reduction inside;
- readback: the vertex fields materialised or reduced by `measurement.reduce` (measure `sum`, stored under the energy's name), the shares derived at read.

### 3.1 `signal.field_form` phase current: a discretisation option (extend, not duplicate)

`phase_current` already exists as a form of `signal.field_form` (the vertex gradient, the reference: nxr core_phase). The intrinsic-edge discretisation of the same quantity becomes the
option `discretisation` of that form (default `vertex_gradient`), never a new node or a new name:
- same ports and types (x complex surface field → y vector3), same measures (`phase_current_*`), so nothing downstream changes;
- the option goes into the provenance and the cache key (the method rule);
- options of the complex (`mollify`, `flip_eps`) default to `eigen solve`'s (1e-6, 1e-6), so the complex is the one the modes of that surface are solved on;
- `signal.gradient_power` is not extended: |∇z|² on the edges has no use yet; it can take the same option when one appears.

This needs no type change: the edges live inside the node.

### 3.2 `signal.phase_split`

Decided 2026-11-19 (PI, on #140): the outputs are named by the mathematics (exact: curl-free; coexact: divergence-free), the physical reading lives in the reference. The parameter `one_form: phase_gradient | phase_current`
(default `phase_gradient`, the reference) chooses the 1-form that is split; it is a step parameter (provenance, cache key), other values are refused. The provenance record holds `boundary`
(`none` | `neumann_potential_dirichlet_stream`: Neumann potential and Dirichlet stream function on an open surface), a fact of the method, not an option. The phase-wrap limit is stated in the reference only.

A new node (no existing node computes it). Its input `modes` is the eigenmode package bound by the runner as it is for a spatial band (`runner.cpp`, the `hodge` constant: P, mu,
kind, positions and faces of the surface the modes are solved on). The field must be on that surface: on a two-piece cortex, the glued surface (the kernel applied with `glued`, as
`meeg.apply_kernel` already does); on a one-piece surface, the surface itself. A field on another surface is refused at plan with its reason.

## 4. The blocker: the package's 2-form parts

`splitPhase` needs the modes' **2-form parts X2 [nF x K]** (the coexact modes' 1-forms are ⋆1⁻¹ d1ᵀ ⋆2 X2) and the complex they were solved on. `eigen solve` stores both
(`<stem>_forms2`, an operator from the mode axis to the complex's faces; the complex with d0, d1, stars, signposts), but the runner binds only the eigenvectors (P) of a package.
Three ways to get X2 to the node:

| | how | type system | cost |
|---|---|---|---|
| A | `forms2` as its own operator input of the node | needs a FACE kind: its target is the complex's faces, today typed as `points` | a type-system change (a `cells` kind with a dimension, or edge/face kinds) |
| B | the package constant carries its 2-form parts: the runner, which already reads the eigenvectors and the glued surface of a package, also reads `<stem>_forms2` into the same constant (a second array of the constant, not a port) and the Delaunay options of the complex (from the package's provenance) | none: the package stays one operator input typed by its eigenvectors | a change to how the runner binds a package (one place, `runner.cpp`); the graph's `Array` holds one matrix today, so the constant needs a second array (or the node reads a second constant the runner adds under the same input) |
| C | rebuild X2 in `prepare` from the complex and a fresh eigensolve | none | a solve per run: breaks "eigenmodes first" (stored modes are used, never re-solved) and wastes time. Rejected |

**Chosen: B** (the package's full forms are part of the package constant: `meta.forms2` the stored float32 bytes, `meta.forms2_shape`, `meta.complex` the options; the
complex rebuilt by `intrinsicComplex(surface, star1Floor)`, its ⋆1 raised as the pencil's). **Recommendation was B.** It keeps edges and faces out of the type system (the PI's direction), and the package is already bound as one constant; it only adds its 2-form parts.
A is the right long-term answer if 1-forms and 2-forms are ever to be stored as Fields or passed between nodes (a `cells` manifold kind with the cell dimension), and is a
type-system change the PI must decide.

Tests: `test_phase_split` (the node through dispatch on a sphere with its modes; equal to the compute functions; independent of the blocks; shares from partition sums;
refusals), `test_hodge_power` (the node through the runner on a solved package), `test_runner_execute` and `test_runner_plan` (the discretisation option).

## 5. Tests (when built)

Synthetic only (an icosphere, a disc), each node through the graph's dispatch point as `pipelines/test/test_member_analytic.cpp` does:
- phase current: on a fine disc, `discretisation: intrinsic_edges` agrees with `vertex_gradient` to the discretisation error, and equals `sharp(c, phaseCurrent(c, z))` exactly;
  the result does not depend on the chunk size;
- phase split: on a sphere with its modes solved in the test, a pure exact field gives `exact_energy` = `one_form_energy` summed over every member of a partition and
  `coexact_energy` ≈ 0; the derived shares from `measurement.reduce` sums equal the whole-complex ratios; a field on another triangulation is refused.

## 6. Not decided here

- Whether the coefficients become an output (a made line with role `modes` per sample) or stay a compute-only result (Python, `nn.compute.signal.projectOneForm`).
- Whether a cross-spectrum's phase 1-form (arg T on the edges, from a stored cross-spectrum) enters the split: that input is an edge quantity and needs either A or a node
  that reads the stored cross-spectrum and forms w inside (the node, not a port, holds the edges).
