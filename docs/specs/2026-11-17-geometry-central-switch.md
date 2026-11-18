# Discrete geometry on geometry-central: one intrinsic triangulation, one gauge (spec, 2026-11-17)

Status: **design, no code** (the inventory script, `tools/geometry_inventory.py`, aside). It specifies how nxr-neuro's discrete geometry moves
entirely onto geometry-central (vendored, `compute/deps/geometry-central`): the intrinsic Delaunay triangulation, the tangent spaces, transport, the
connection and cotan Laplacians, the masses and DEC. nxr-neuro's own line port of the Studio's intrinsic Delaunay construction
(`compute/include/nxr/delaunay.h`, `compute/src/delaunay.cpp`) and the charts built on it (`nxr/chart.h`) are retired. Section 10 lists the open
questions, each with a recommendation; appendix A is the generated inventory.

Every geometry-central name below is spelled as in its sources at the pinned commit (`8579865`, `v1.0.0-19-g8579865`).

## 0. Decision (maintainers, 2026-11-17)

1. nxr-neuro switches fully to geometry-central's implementation: its intrinsic triangulation with signposts (section 2.2 compares
   `SignpostIntrinsicTriangulation` with `IntegerCoordinatesIntrinsicTriangulation` and recommends the first), its tangent spaces
   (`halfedgeVectorsInVertex`, `vertexTangentBasis`), its transport (`transportVectorsAlongHalfedge`), its connection Laplacian, cotan Laplacian,
   masses and DEC.
2. **One gauge everywhere: geometry-central's.** The canonical frame (north / west, from the registered sphere) is stored as an angle array
   relative to that gauge.
3. Exact numerical parity with the Studio's TypeScript, nxr and Rheome is **no longer required**: those implementations are being phased out, and
   the Studio will run nxr-neuro. Correctness is shown against analytic results (section 4), and one comparison with the old port explains every
   difference once.

## 1. Why: two implementations, two gauges

A tangent vector at a vertex is stored as a complex number z = a + ib in that vertex's tangent space. The **gauge** is the choice, at every vertex,
of the direction called angle 0 and of how the angles of the edges around the vertex are measured. Two gauges coexist in `compute/` today.

| | the port's gauge (`nxr/delaunay.h`, `nxr/chart.h`, `geometry::vertexFrames`) | geometry-central's gauge (`IntrinsicGeometryInterface`, `EmbeddedGeometryInterface`) |
|---|---|---|
| reference halfedge (angle 0) | the vertex's **first** outgoing halfedge in face order (the lowest face that lists it); at a boundary, its twinless outgoing halfedge (`addSignposts`) | `v.halfedge()`: as `ManifoldSurfaceMesh` is built from a face list, the outgoing halfedge in the **last** face that lists the vertex; at a boundary, the interior halfedge that starts the counter-clockwise arc |
| angles as stored | the raw corner angles summed, in [0, Θ_v) (`Mesh::angle`); carried through flips modulo the cone angle Θ_v | `SignpostIntrinsicTriangulation::signpostAngle`: also raw, in [0, Θ_v), "not rescaled" (its header) |
| angles as used | **rescaled at use** by `rescaledAngle`: ψ = 2π · angle / Θ_v at an interior vertex; at a boundary vertex **not rescaled** (cone target = the vertex's own angle sum) | rescaled in `cornerScaledAngles` and `SignpostIntrinsicTriangulation::halfedgeVector` (`vertexAngleScaling`): to **2π** at an interior vertex, to **π** at a boundary vertex |
| 3D realisation of angle 0 | e1 = the reference edge projected on the tangent plane (`chart::tangentBasis`, float32 values; `geometry::vertexFrames`, double) | e1 = the **average** of all outgoing edges, each projected and rotated back by its own rescaled angle (`computeVertexTangentBasis`) |
| normal | area-weighted face normals (`frames::vertexNormals`, `vertexFrames`) | corner-angle-weighted face normals (`computeVertexNormals`) |
| transport across an edge | `connectionLaplacian`: e^{i n (ψ_h + π − ψ_rev)} from the rescaled signposts | `transportVectorsAlongHalfedge`: `unit(-vecB / vecA)` from `halfedgeVectorsInVertex` |

Verified in the sources: the port's angles **are** rescaled to 2π, but only at use (`rescaledAngle`, `reverseAngle`), never in storage, and never at
a boundary vertex. Both gauges are therefore legitimate discretisations of the same Levi-Civita connection; they differ by a rotation e^{iθ_v} per
vertex (the two choices of angle 0) plus, at boundary vertices, a different rescaling. The rotation is invisible in any gauge-invariant result (an
eigenvalue, a speed, a divergence) and fatal in anything that mixes coordinates from the two (a complex eigenmode decoded in the other frame, a
vector-heat transport compared with a connection-Laplacian solution).

**Who works in which gauge today** (the inventory, appendix A, lists every file):
- **the port's gauge:** the vertex connection Laplacian (`connection_laplacian.cpp` → `delaunay::connectionLaplacian`), `geometry::vertexFrames` and
  `vertexGrid` (what its complex eigenvectors decode against), the charts (`chart::ChartKit`: heat distance, `transported`, `logMap`,
  `gaugeHeading`), the frames that read them (`frames::gaugeAngle`, `angleFromNorth`, the stored `<surface>_canonical_gauge_angle`), the ladder's
  Levi-Civita transport (`ladder::leviCivita`), quadrature, crossings, and the optical flow's connection gauge (`flow::connection::Corners`);
- **geometry-central's gauge**, through five files that use its embedded frames: `vertex_frames.cpp` (`vertexFramesEmbedded`),
  `geometry_grid.cpp` (`vertexGridEmbedded`), `geometry_bundle.cpp` (the light Geometry bundle's `vertexGrid`), `facets.cpp` (the trivial gauge
  combs `vertexGridEmbedded`) and `dirac_operator.cpp`; and through `vertexTangentBasis` / `transportVectorsAlongHalfedge` directly in
  `vector_heat.cpp` (the vector heat method, **on the input mesh**, not the intrinsic one), `direction_field.cpp`, `smooth_field.cpp`, `gauge.cpp`
  (the trivial connection) and `curvatures.cpp`.

A test through the MATLAB binding exposed the mismatch: connection-Laplacian modes decoded with one grid and vector-heat results with the other.

## 2. The target

### 2.1 One intrinsic surface object

A thin class in `compute/` (working name `nxr::manifold::IntrinsicSurface`, header `nxr/intrinsic_surface.h`) owns the input mesh, its geometry and
the intrinsic triangulation, and is the only place the construction is written. It is pure (arrays in, the object out) and keeps the port's record.

Construction, in this order (each step names the geometry-central call):
1. `makeManifoldSurfaceMeshAndGeometry` (or the `ManifoldSurfaceMesh` + `VertexPositionGeometry` the `Manifold` already holds): the input mesh and
   its embedded geometry.
2. An `EdgeLengthGeometry` of the input mesh with the input's `edgeLengths`, then `mollifyIntrinsic(mesh, edgeLengths, mollify)` on **those
   input lengths** (option `mollify`, section 3). ε is recorded as the length added (the difference before and after; 0 when nothing was needed).
3. The mass of the mollified **input** is recorded here, before any flip, when the mass option asks for it (section 3.3).
4. `SignpostIntrinsicTriangulation(inputMesh, mollifiedInputGeometry)`: the signposts are laid out from the mollified input, so angle 0 at each
   vertex is the input's `v.halfedge()` and stays so through every flip. nxr subclasses it only to override `computeVertexDualAreas` (the mass
   option, section 3.3); nothing else is overridden.
5. `triangleTestEPS = flipEps` (option, section 3.2), then `flipToDelaunay()`. The flips are counted through `edgeFlipCallbackList` (geometry-central
   keeps no count); `isDelaunay()` is checked after.
6. The record: `mollify`, ε, `flipEps`, flips, the mass option, the geometry-central version (section 7), and the gauge identity (section 5.1).

With the named option `mesh = embedded` (today's `MeshKind::Embedded`), steps 2 to 5 are skipped and every operator is the input geometry's.

Each connected piece of a surface is one `IntrinsicSurface` (as `ChartKit` solves each piece by itself today), so the heat and vector heat solvers
never see a disconnected Laplacian; values off a piece stay NaN as now.

### 2.2 Which intrinsic triangulation: signposts or integer coordinates

| | `SignpostIntrinsicTriangulation` | `IntegerCoordinatesIntrinsicTriangulation` |
|---|---|---|
| tangent spaces through flips | `halfedgeVectorsInVertex` is **overridden** (`computeHalfedgeVectorsInVertex`) to read `signpostAngle`, which is measured from the **input** `v.halfedge()`: the gauge is the input mesh's, before and after flips, and equals the one `vertexTangentBasis` realises | `updateHalfedgeVectorsInVertex` re-derives the angles from the **intrinsic** `v.halfedge()`, which a flip may move (`SurfaceMesh::flip` reassigns it): the gauge drifts away from the input's `vertexTangentBasis`, and a per-vertex correction angle would have to be tracked |
| mollification | not built in: done on the input lengths before construction (step 2 above), so the signposts see the mollified lengths | built into the constructor (`mollifyEPS = 1e-5` by default, silently, its own TODO says), applied **after** the normal coordinates are set |
| flip rule | `flipEdgeIfNotDelaunay`: flips while `edgeCotanWeight(e) <= -triangleTestEPS`, refuses a fixed or boundary edge and a non-finite new length | the same Delaunay test through `isDelaunay(e)` |
| what it is better at | cheap, no combinatorial bookkeeping; tracing can fail numerically on badly degenerate input | robust tracing and common subdivision, vertex insertion and refinement (`delaunayRefine`, `insertCircumcenterOrSplitSegment`) |
| what nxr-neuro needs | flips only (no insertion: the vertices are the recording's source points), tangent spaces in one gauge, no tracing on the hot path | not needed |

**Recommendation: `SignpostIntrinsicTriangulation`.** Its tangent spaces are the input's by construction, which is exactly "one gauge
everywhere"; integer coordinates would need nxr to track a gauge correction per vertex, i.e. to rebuild what signposts give. If refinement or exact
tracing is ever needed (e.g. a common subdivision to draw intrinsic edges), an `IntegerCoordinatesIntrinsicTriangulation` can be built beside it for
that purpose only, never for tangent quantities.

### 2.3 The one gauge

**The gauge of nxr-neuro is geometry-central's vertex gauge of the (mollified) input mesh:** angle 0 along the input's `v.halfedge()`, corner
angles rescaled by `cornerScaledAngles` (2π, or π at a boundary vertex), and its 3D realisation `vertexTangentBasis` with the normal
`vertexNormals`. On the intrinsic triangulation the same gauge is read from `SignpostIntrinsicTriangulation::halfedgeVectorsInVertex`, and transport
from its `transportVectorsAlongHalfedge`. Every complex tangent value in nxr-neuro is in this gauge, and every frame is either this one (the
`vertexGrid`) or an angle array relative to it (the canonical frame, section 6).

When ε > 0, the input's 3D frame must be realised from the **mollified** angles, which `VertexPositionGeometry::vertexTangentBasis` (computed from
the unmollified positions) does not do. The frame is then computed with `computeVertexTangentBasis`'s own formula (the average of the outgoing
edges, projected and rotated back by `halfedgeVectorsInVertex[he].arg()`) over the mollified input geometry's `halfedgeVectorsInVertex`: a dozen
lines in `IntrinsicSurface`, with a test that it equals `vertexTangentBasis` to 1e-12 when ε = 0 (section 10, question 5).

### 2.4 What replaces what

| today (port) | geometry-central | notes |
|---|---|---|
| `delaunay::makeMesh`, `Mesh` | `ManifoldSurfaceMesh`, `EdgeLengthGeometry`, `SignpostIntrinsicTriangulation` | the halfedge numbering changes: anything stored per intrinsic edge or face records the triangulation (section 5.1) |
| `mollify` | `mollifyIntrinsic(mesh, edgeLengths, relativeFactor)` | the same formula (δ = factor × mean edge; ε = max over interior halfedges of l_C − l_A − l_B + δ, from 0; added to every edge). One difference: `std::fmax` ignores a NaN, the port's JavaScript-style max propagates it; a NaN length is refused before construction instead |
| `addSignposts` | the `SignpostIntrinsicTriangulation` constructor (`signpostAngle`) | reference halfedge changes (section 1) |
| `flipToDelaunay`, `flip` | `IntrinsicTriangulation::flipToDelaunay`, `flipEdgeIfNotDelaunay` | the same FIFO queue of every edge and the same four neighbours queued; flips counted by callback |
| `cotanLaplacian`, `edgeCotanWeights`, `faceAreas` | `cotanLaplacian`, `edgeCotanWeights`, `faceAreas` (`require…`) | positive semidefinite in both |
| `galerkinMass` | `vertexGalerkinMassMatrix` | |
| `barycentricMass` | `vertexLumpedMassMatrix` / `vertexDualAreas` | geometry-central's "lumped" mass **is** barycentric (A/3) |
| `mixedVoronoiMass`, `Mesh::mass` | **none**: rebuilt on geometry-central (Meyer et al. 2003 from the input geometry's `cornerAngles`, `edgeLengths`, `faceAreas`) and served through the `computeVertexDualAreas` override | section 3.3 |
| `rescaledAngle`, `reverseAngle` | `halfedgeVectorsInVertex[he].arg()`, `halfedgeVectorsInVertex[he.twin()].arg()` | |
| `connectionLaplacian(m, 1)` | `vertexConnectionLaplacian` | |
| `connectionLaplacian(m, n)`, n > 1 | none: rebuilt from `transportVectorsAlongHalfedge[he].pow(n)` and `edgeCotanWeights`, as geometry-central's own `direction_fields.cpp` does | the n-RoSy bundle |
| `dec` (d0, d1, ⋆1, ⋆2) | `requireDECOperators`: `d0`, `d1`, `hodge1`, `hodge2`; `hodge0` is `vertexDualAreas`, i.e. the chosen mass through the override | the edge orientation is `e.halfedge()` (tail → tip) in both; the edges are numbered differently |
| Hodge–Dirac (`hodge_dirac.cpp`), `intrinsic_forms.cpp` | none: **rebuilt on geometry-central's triangulation** from its DEC operators and the chosen mass | the maths unchanged, only its inputs |
| `nonDelaunayEdges`, `triangleArea`, `cornerAngle`, `halfCot`, `edgeWeight` | `isDelaunay(e)`; `faceArea`, `cornerAngle`, `halfedgeCotanWeight`, `edgeCotanWeight` (immediates) | |
| `chart::HeatMethod` | `HeatMethodDistanceSolver` on the `IntrinsicSurface` | it reads `vertexLumpedMassMatrix`, so the mass option reaches it through the override; `tCoef` the same |
| `ChartKit::transported`, `logMap` | `VectorHeatMethodSolver::transportTangentVector`, `computeLogMap` (`LogMapStrategy::VectorHeat`) on the `IntrinsicSurface` | `vector_heat.cpp`'s solver moves from the input mesh to the intrinsic one (section 3.4) |
| `chart::tangentBasis`, `vertexFrames`, `vertexGrid` | `vertexTangentBasis`, `vertexNormals` (one frame; `vertexFramesEmbedded` / `vertexGridEmbedded` become the only implementation of `vertexFrames` / `vertexGrid`) | the float32 rounding the Studio's basis had is dropped |
| `ChartKit::gaugeHeading`, `headingVector` | `atan2` of a vector against `vertexTangentBasis`, and its inverse | one line each |
| `flow::connection::Corners` (the optical flow's face-by-face gauge) | rebuilt on the mollified input geometry's `halfedgeVectorsInVertex` and `cornerScaledAngles` | same gauge as the connection Laplacian, by construction |

## 3. Method choices: named options, defaults stated

The port's deliberate choices stay explicit: each is a named, recorded option of `IntrinsicSurface` (and of `ManifoldOptions`), reproduced through
geometry-central where possible.

| option | default | through geometry-central | reproduced? |
|---|---|---|---|
| `mesh` = `intrinsic` \| `embedded` | `intrinsic` | `SignpostIntrinsicTriangulation` \| the input `VertexPositionGeometry` | yes |
| `mollify` (δ relative to the mean edge length) | `1e-6` | `mollifyIntrinsic(…, relativeFactor)` on the input lengths before construction | **yes, exactly** (same formula; NaN refused first) |
| `flipEps` (flip while the cotan weight is below −flipEps) | `1e-6` | `IntrinsicTriangulation::triangleTestEPS` | **yes**, with two notes below |
| `mass` = `input-mixed-voronoi` \| `barycentric` \| `galerkin` | `input-mixed-voronoi` | override of `computeVertexDualAreas` \| geometry-central's own \| `vertexGalerkinMassMatrix` | **yes, through geometry-central's extension point**; the mixed-Voronoi formula itself is rebuilt |
| boundary cone target | geometry-central's: π | `cornerScaledAngles`, `vertexAngleScaling` | **no**: the port's (no rescaling at a boundary) is not reproduced; recommendation below |
| heat time `tCoef` | `1.0` (t = tCoef × mean edge²) | the solvers' `tCoef` | yes, same definition |

### 3.1 The mollification δ (`mollify`, Sharp & Crane 2020)

Reproduced exactly. With δ = `mollify` × mean edge, ε = max(0, max over corners of l_C − l_A − l_B + δ) is added to every edge. On a mesh whose
triangles are all comfortably valid, ε = 0 and nothing changes. **Consequence for eigenvalues:** when ε > 0 every length grows by ε, the metric by a
relative amount of order ε / mean edge (at most of order δ on a mesh with a near-degenerate triangle, 10⁻⁶), and the Laplacian's eigenvalues
shrink by about twice that: far below the discretisation error, but not zero; it is recorded. `mollify = 0` adds no margin; ε is still added where a
triangle violates the triangle inequality. The mollification must happen **before** the signpost triangulation is built (step 2 of 2.1):
mollifying the triangulation's `edgeLengths` afterwards would leave signposts laid out from the unmollified angles.

### 3.2 The flip tolerance (`flipEps`)

Reproduced: `flipEdgeIfNotDelaunay` returns without flipping while `edgeCotanWeight(e) > -triangleTestEPS`, the port's rule (flip below
−flipEps; the two differ only at exact equality). Notes:
- `triangleTestEPS` is shared: geometry-central also uses it in `flipEdgeIfPossible`'s area test and in insertions. nxr-neuro calls neither, so
  setting it to `flipEps` changes nothing else; the record names it.
- The refusals differ slightly: the port also refuses a flip whose new length is ≤ 0; geometry-central refuses only a non-finite one. Both allow
  self-edges (`flip(e, false)`) and refuse a boundary edge and a degree-1 vertex. A test asserts every length is positive after the flips.
- **Consequence for eigenvalues:** edges with a cotan weight in (−flipEps, 0) are left unflipped, so the cotan Laplacian may keep negative
  off-diagonal entries of size up to flipEps × the local scale. Its eigenvalues move by at most that order (relative 10⁻⁶ for the default), and the
  discrete maximum principle may fail at that level. `flipEps = 0` flips every non-Delaunay edge (ties of cocircular quads are left; their weight
  is 0, so the Laplacian is unique, Bobenko & Springborn 2007).

### 3.3 The mass: the input's mixed Voronoi, recorded before any flip

The port's default is the mixed-Voronoi area (Meyer, Desbrun, Schröder & Barr 2003) of the **mollified input** triangulation, recorded before the
first flip, paired with the cotan Laplacian of the **intrinsic** Delaunay triangulation. geometry-central has no mixed-Voronoi mass, and its
solvers (`HeatMethodDistanceSolver`, `VectorHeatMethodSolver`) and its `hodge0` read `vertexLumpedMassMatrix` / `vertexDualAreas` of the geometry
they are given, which is barycentric on the intrinsic mesh.

It is reproduced through geometry-central's extension point: `computeVertexDualAreas` is virtual, and the `IntrinsicSurface` subclass overrides it
to return the chosen mass. Then `vertexDualAreas`, `vertexLumpedMassMatrix`, `hodge0`, the heat method and the vector heat method all use the same
mass, with no other change. The mixed-Voronoi formula is written once in nxr (from the input geometry's `cornerAngles`, `edgeLengths` and
`faceAreas`), the one piece of DEC that is "rebuilt on geometry-central's triangulation".

Options: `input-mixed-voronoi` (default, as now), `barycentric` (geometry-central's own: A/3 of the intrinsic triangles), `galerkin`
(`vertexGalerkinMassMatrix`, a sparse matrix; the eigen solver takes it, the diagonal-mass solvers refuse it with a message).

**Consequence for eigenvalues.** The generalised eigenproblem L φ = λ M φ depends on M:
- `galerkin` (consistent P1 mass): the conforming finite-element pair, whose eigenvalues are upper bounds of the continuum ones (Rayleigh–Ritz)
  and converge as O(h²).
- a diagonal (lumped) mass gives eigenvalues usually **below** the Galerkin ones, also O(h²). `barycentric` and `input-mixed-voronoi` have the same
  total area (flips preserve area) but distribute it differently: on an irregular mesh the eigenvalues differ at O(h²), and the splitting of the
  sphere's degenerate clusters (2l + 1) differs.
- `input-mixed-voronoi` pairs a mass of the input triangulation with a stiffness of the intrinsic one; both discretise the same surface with the
  same vertices, so this is consistent to O(h²), and it keeps the mass a vertex carries independent of which edges were flipped (a reason the port
  chose it: masses that do not move under a flip).
The analytic tests (section 4) measure each option's error on the sphere and under refinement, so the default is kept on evidence, not habit.

### 3.4 What geometry-central does not reproduce, and the recommendation

- **The boundary cone target.** geometry-central rescales a boundary vertex's angles to π; the port did not rescale them. On a flat patch with a
  boundary, the port's transport is exactly flat at boundary vertices and geometry-central's is not (a boundary corner of 90° is stretched to 180°).
  `vertexAngleScaling` is not virtual, and overriding `computeHalfedgeVectorsInVertex` would fight `updateAngleFromCWNeighor`, which writes the same
  member during flips. **Recommendation: accept geometry-central's convention.** Cortical hemispheres are closed surfaces (no boundary vertex); a
  surface with a boundary (a cut medial wall, a patch) records `boundary vertices: N` in its record, and the flat-patch test (section 4) checks
  interior vertices exactly and documents the boundary behaviour.
- **The NaN semantics of `mollify`** (section 2.4): refused before construction instead.
- **The vector heat method on the input mesh** (`vector_heat.cpp` today) is not a choice the port made; it moves to the intrinsic triangulation, so
  that every tangent operator is on the same mesh.
- **The normal.** The port's area-weighted normal is replaced by geometry-central's corner-angle-weighted `vertexNormals`, the one tangent plane
  every frame lies in (section 10, question 3).

## 4. Verification against analytic results

New compute tests (scoped through `tools/tests.py`, synthetic meshes generated in the test; no golden files needed). Each runs on the
`IntrinsicSurface` with every mass option, and on a refinement sequence (icospheres of levels 2 to 6; the same spheres with vertices jittered
tangentially, which makes many edges non-Delaunay; a UV sphere with slivers at its poles). Tolerances are stated as convergence rates, not fixed
numbers.

| test | analytic result | what is checked |
|---|---|---|
| A1 sphere Laplacian | eigenvalues of the unit sphere: l(l + 1), multiplicity 2l + 1 | the first 7 clusters (l ≤ 6): cluster sizes exactly 2l + 1 (gaps between clusters ≫ spread within); relative error O(h²) (slope ≥ 1.8 in log–log); `galerkin` errors positive (upper bounds); `mesh = embedded` on the jittered spheres is worse than `intrinsic` (the reason for the intrinsic triangulation) |
| A2 sphere connection Laplacian | the rough Laplacian ∇*∇ on tangent vector fields of the unit sphere (spin-1 harmonics): l(l + 1) − 1 for l ≥ 1, complex multiplicity 2l + 1 (3, 5, 7, 9 …); for the n-RoSy bundle, l(l + 1) − n² for l ≥ n | `vertexConnectionLaplacian` and the n > 1 build; no zero eigenvalue on the sphere (no parallel field: Gauss–Bonnet); O(h²) convergence |
| A3 gauge invariance | the spectrum of the connection Laplacian does not depend on the gauge | a random diagonal rotation D = diag(e^{iθ_v}) of the gauge: D K D^H has the same eigenvalues; eigenvectors map by D |
| A4 flat patch | on a planar mesh the Levi-Civita connection is flat: transport along any edge maps the embedded vector to itself; holonomy around every interior vertex is 0 | a jittered planar disc: for every interior edge, the 3D vector realised by `vertexTangentBasis` at the tail, transported by `transportVectorsAlongHalfedge`, realises the same 3D vector at the tip (1e-12); a constant field is in the kernel of the connection Laplacian at interior vertices; `transportTangentVector` from one vertex returns the constant field; `computeLogMap` equals the planar coordinates (O(h)); the boundary behaviour (section 3.4) asserted as documented |
| A5 holonomy and Gauss–Bonnet | the angle defects sum to 2πχ; the holonomies of the vertex connection around the triangles (its elementary cycles, each taken in (−π, π]) sum to 2πχ | sphere (χ = 2), torus (χ = 0), double torus (χ = −2); exact to rounding |
| A6 Hodge decomposition of known fields | on the unit sphere, X = ∇f + J∇g with f, g spherical harmonics (e.g. f = Y₁⁰ = z, g = Y₂¹ ∝ xz): the exact part is ∇f, the co-exact part J∇g, the harmonic part 0; on a torus the harmonic 1-forms have dimension 2 (= 2g) | DEC (`d0`, `d1`, `hodge1`, `hodge2`, the chosen mass): the parts recovered with O(h) error or better; the kernel dimension of the Hodge Laplacian on 1-forms exactly 0 (sphere), 2 (torus), 4 (double torus) |
| A7 Hodge–Dirac | D² is the Hodge Laplacian; on the unit sphere D's nonzero eigenvalues are ±√(l(l + 1)), and its kernel is the harmonic 0- and 2-forms (dimension 2) | the rebuilt Hodge–Dirac on geometry-central's DEC |
| A8 geodesic distance and log map | on the unit sphere the distance from a pole is the colatitude; the log map's radius is the colatitude and its angle the longitude (up to the gauge's constant offset at the source) | `HeatMethodDistanceSolver`, `computeLogMap`; O(h) |
| A9 gauge realisation | `vertexTangentBasis` is orthonormal, tangent to `vertexNormals`, and `halfedgeVectorsInVertex[v.halfedge()]` has angle 0 | every mesh; plus the ε > 0 realisation of 2.3 equal to `vertexTangentBasis` when ε = 0 |
| A10 the record | mollify, ε, flipEps, flips (counted), isDelaunay after, the mass option, the version | every mesh |

**The one-time comparison with the old port** (one PR, section 9, G2; a test kept until the port is deleted; its numbers go into appendix B of this
spec when it lands). On the same synthetic meshes it builds both, and explains every difference:
1. **The gauge rotation.** For each vertex, θ_v = the angle of the port's reference halfedge in geometry-central's gauge
   (`halfedgeVectorsInVertex` of that halfedge). Then K_gc = D K_port D^H with D = diag(e^{iθ_v}) (or its conjugate, by the transport convention,
   which the test fixes once) to rounding at every interior vertex. Any residual is attributed to one of the items below and must vanish when
   that item is neutralised.
2. **The triangulation.** Both flip to an intrinsic Delaunay triangulation; the order differs (edge numbering), so with `flipEps > 0` the two may
   leave different near-Delaunay edges unflipped: the cotan Laplacians differ by O(flipEps). With `flipEps = 0` they must be equal to rounding
   (the intrinsic Delaunay Laplacian is unique).
3. **Boundary vertices**: the π rescaling; reported per vertex.
4. **The masses**: the mixed-Voronoi mass equal to rounding (same formula, same mollified input); barycentric and Galerkin equal to rounding
   when the triangulations are equal.
5. **The frames**: the angle between the port's e1 (reference edge projected, area-weighted normal, float32) and geometry-central's
   `vertexTangentBasis` (averaged, angle-weighted normal), after removing θ_v: its distribution reported; it is the 3D realisation difference only.
6. **Charts**: heat distance and log map radius equal to rounding (gauge-free, same Laplacian and mass); the log map angle and `transported` equal
   after the rotation by θ_v.

## 5. Stored data and the gauge

### 5.1 The gauge is recorded

**Terms.** *Gauge* means the discretisation's tangent-space convention (section 2.3); *frame* means a reference a measurement is reported in
(canonical north / west, the anatomical projection). Rows that today say `"gauge": "canonical frames"` (`measure.cpp`, `tiles.cpp`) are renamed
`"frame": "canonical"`; `"gauge"` then always names a gauge.

**The gauge record.** Every stored tangent quantity (complex eigenmodes of the connection Laplacian, complex or 2-component vector fields, transported
frames, the canonical gauge angle, the tiles' Levi-Civita form) carries in its row and its provenance:

```
"gauge": "geometry-central@8579865/vertex",           // library, pinned commit, convention (vertex tangent spaces of the input mesh)
"gauge_surface": "<the surface's id>",               // the gauge is a function of the surface's connectivity and face order
"gauge_record": {"mollify": 1e-6, "epsilon": 0, "flipEps": 1e-6, "mass": "input-mixed-voronoi", "boundary_vertices": 0, "fingerprint": "<8 hex>"}
```

The fingerprint is a hash of each vertex's reference neighbour (the head of `v.halfedge()`) and ε: two values are in the same gauge exactly when
`gauge`, `gauge_surface` and the fingerprint agree. Anything stored per **intrinsic** edge or face (a 1-form on the intrinsic triangulation, ⋆1)
records the same block, since edge numbering and orientation are the triangulation's.

**Stored as 3D where possible.** A tangent vector field is stored as an embedded 3-vector per vertex (`ValueType::vector3`, tangent to the surface),
which needs no gauge at all; the complex form is stored only where the complex structure is the result (eigenmodes of the connection Laplacian,
n-RoSy fields, spin-weighted quantities). The optical-flow velocity is already stored embedded; it stays so.

**Old-gauge data.** A stored tangent quantity without a `gauge` block, or with another one, is **refused** when read as a tangent quantity, with a
message naming what to do:

```
<field> is stored in the port's signpost gauge (no gauge record); nxr-neuro now uses geometry-central@8579865/vertex.
Recompute it: nxr-neuro run <pipeline> … (development data may be recomputed; the old arrays are kept until removed with the format's tools).
```

Development data is recomputed (a one-off run per affected pipeline). Gauge-free quantities (scalars, eigenvalues, 3D vectors, speeds,
divergence, curl) are untouched.

### 5.2 Ports and the executor carry the gauge

`nxr::graph::PortType` (`pipelines/graph/include/nxr/graph/types.h`) gains two members, filled by the runner from the gauge record and by a node's
plan from its inputs:

```
int spin = 0;            // 0 scalar; 1 a tangent vector; n an n-RoSy (line field: 2); -1 a conjugate (co-vector in complex form)
std::string gauge;       // "" for spin 0 and for vector3 values; else the gauge id (gauge + gauge_surface + fingerprint)
```

`check` (the arrow check, before any data flows) refuses an arrow whose two ends have different `spin`, or the same nonzero spin and different
`gauge`, naming both and the conversion to insert (`geometry.frame_rotate`). A node declares, per port, either a fixed spin and "same gauge as input X"
(covariant: most tangent nodes) or a produced gauge (a node that builds the operators: its gauge is the bound `IntrinsicSurface`'s). Bound
constants (mesh operators) carry the gauge of the surface they were built on, so an operator from one surface can never act on a field in another
surface's gauge. `compute` on arrays handed to it takes the gauge from the caller's metadata (or the sidecar), and refuses a complex tangent array
that names none.

### 5.3 The conversion node: `geometry.frame_rotate`

A pure, pointwise node (traits: `pointwise`, precision follows the input):

```
geometry.frame_rotate(field, to, spin) -> field'
  field  [V] complex, spin k, in gauge g
  to     [V] real, radians: the angle of the target reference measured in g (e.g. <surface>_canonical_gauge_angle, north's angle in the gauge)
  spin   k (from the field's type; a parameter only when the input names none)
  field'_v = field_v · e^(−i k to_v)     its port: frame = to's frame (e.g. "canonical"), same surface
```

With `to` = north's gauge angle, a spin-1 field becomes components on (north, west); with spin 2 a line field's double angle is measured from
north. Its siblings, also pure: `geometry.frame_embed(field, grid) -> vector3` (z → a e1 + b e2) and `geometry.frame_project(vector3, grid) -> complex` (its inverse
on the tangent plane). A rotation between the old port's gauge and the new one is never shipped as a node: old data is recomputed (5.1).

## 6. Frames (`compute/include/nxr/frames.h`)

The canonical north / west is kept and stored as **one angle array relative to the gauge**; the anatomical anterior / superior becomes a
**reporting projection**, not a frame.

What changes in `frames.h`:
- `Canonical`, `canonicalFrames`: unchanged maths (the registered sphere's meridian pushed forward by the 2×2 least-squares Jacobian, the
  singular vertices); the normal it projects on is geometry-central's `vertexNormals` (passed in, as now).
- `vertexNormals` (area-weighted): **removed**; callers take `vertexNormals` of the input geometry (the one tangent plane).
- `gaugeAngle(chart::ChartKit&, direction)` → `gaugeAngle(const IntrinsicSurface&, direction)`: atan2 against `vertexTangentBasis`. The canonical
  frame's stored form is `gaugeAngle(surface, e1)`, the field `<surface>_canonical_gauge_angle` (kept name), now with its gauge record.
- `angleFromNorth(kit, vectors, e1)` → **removed** from `frames.h`; it is `geometry.frame_rotate(field, canonical_gauge_angle, 1)` followed by `arg` (a
  pipeline step, section 5.3).
- `Anatomical`, `anatomicalFrame`, `carryAnatomical`, `anatomicalComponents`, `angleFromAnterior`: become the **anatomical projection** (rename
  `anatomicalProjection`). A reporting projection of a tangent vector onto the anterior and second axes:
  - definition `scs` (no registration): the axes projected on the tangent plane and dotted with the 3D vector; it needs **no frame and no gauge**,
    so the option "reference = signpost gauge" (`Anatomical::reference`) is removed;
  - definition `template`: the angle from north carried through the registered spheres, as now (gauge-free, because north is);
  - `second`, `handed`, `reliable`, `minProjection` unchanged. It never produces a stored frame or a complex field; it produces the reported
    components (`flow_velocity_anterior`, `flow_velocity_second`) from 3D vectors.
- The include of `nxr/chart.h` goes; `frames.h` includes `nxr/intrinsic_surface.h`.

The `[V,3,3]` field `<surface>_canonical_frames` stays readable as a gauge-free export (the Studio reads it today); it is derived from the angle
array and `vertexGrid` when written (section 10, question 8).

## 7. Pinning geometry-central

- The submodule is pinned at `8579865362abd2185ef97d7ef0dae09338ab8ff1` (`v1.0.0-19-g8579865`, 2025-10-13). `THIRD_PARTY_NOTICES.md` names the
  commit (it says only "submodule" today).
- The build exposes it: CMake reads the submodule's commit at configure time into `nxr::compute::geometryCentralVersion()` ("8579865"); a test
  compares it with `git submodule status`, so a moved submodule without a rebuilt version string fails.
- The gauge id (5.1) includes it, so data computed with another commit is not silently mixed.
- **A gauge fingerprint test**: on three fixed synthetic meshes, the hash of `v.halfedge()` per vertex, `halfedgeVectorsInVertex`,
  `vertexTangentBasis` and the intrinsic triangulation after `flipToDelaunay`, compared with values committed beside the test. It fails on any
  change of convention.
- **Upgrading**: a PR that moves the submodule runs (1) the analytic suite (section 4), (2) the fingerprint test, (3) the WebAssembly build
  (`bindings/scripts/build-wasm.sh`, Emscripten, C++17 as `CMakeLists.txt` requires) and the analytic suite under Node from the wasm build, (4) the
  bindings' tests. If the fingerprint changes, the PR bumps the gauge id and lists which stored quantities must be recomputed; if only the
  version changes, the stored gauge ids are migrated by a recorded equivalence (`gauge_equivalent: [old, new]`) after the fingerprint test proves
  equality.
- A scheduled CI job (a workflow beside `.github/workflows/wasm.yml`) builds against geometry-central's current `master` weekly, report-only, so an upstream change is known before it is
  adopted.

## 8. The Studio's transition (what it must change; nothing is changed here)

The Studio keeps its own TypeScript implementations today (`intrinsicDelaunay.ts`, `heatMethod.ts`, `connectionLaplacian.ts` with
`tangentBasis` / `embedTangent`, `surfaceKit.ts`, `intrinsicGauge.ts`). To stay consistent with nxr-neuro it must:
1. **Stop computing tangent quantities itself.** Take the intrinsic operators, frames (`vertexGrid`), transport, log maps and connection modes from
   nxr-neuro (the wasm binding), never mix its own with nxr-neuro's.
2. **Read and check the gauge record** on every tangent field it loads; refuse (or ask to recompute) a field without one, as nxr-neuro does.
3. **Decode complex fields with `vertexGrid`** from the same surface (and the same gauge id), not with its own `tangentBasis`.
4. **Read the canonical frame as the angle array plus the gauge**, or the gauge-free `[V,3,3]` export; `<surface>_canonical_gauge_angle` values
   it wrote itself (in its signpost gauge) are old-gauge data.
5. **Show anatomical components as a projection** of 3D vectors, not as a frame.
6. **Retire its parity goldens** for nxr-neuro (`gen_chart_golden.mts`, `gen_ladder_golden.mts`, `gen_opticalflow_connection_golden.mts` and the
   delaunay parity reference run its code); the analytic suite replaces them.
7. Expect different numbers for gauge-dependent values (complex modes, log map angles) and O(h²)-level differences elsewhere (normals, frames);
   gauge-invariant results (eigenvalues, distances, speeds) agree to the tolerances of section 4's comparison.

## 9. The PR plan, each keeping main green

Work in flight now: the **runner split** (compute-and-run C5: `pipelines/src/runner*`, `bound.cpp`, `registry.cpp`, `crossings_steps.cpp`,
`spatial_events_steps.cpp`; C4 before it rewrites the steps' lookups, `crossings_steps.cpp` among them) and the **io SQL migration** (record API
steps 6 to 10: `meeg.cpp`, the command line, the wasm-io binding, the tests in two or three PRs by folder). This work lives in `compute/` and its
consumers; it is ordered so the compute PRs run in parallel with both, and the pipeline PRs come after the files they touch have been migrated.

| PR | what | touches | waits for | parallel with |
|---|---|---|---|---|
| G0 | this spec and `tools/geometry_inventory.py` | docs, tools | — | everything |
| G1 | `IntrinsicSurface` beside the port (new files only): construction of 2.1, the options of 3, the mixed-Voronoi rebuild and the dual-area override, the ε > 0 frame realisation, the record; the version pin (7); the analytic suite (4) against it | `compute/` new files, `compute/test/` new tests, `THIRD_PARTY_NOTICES.md`, `compute/CMakeLists.txt` (version string) | — | C4, C5, migration 6 to 10 |
| G2 | the one-time comparison (4) as a test, and appendix B of this spec with its numbers | `compute/test/` new test, this spec | G1 | as G1 |
| G3 | `Manifold` on `IntrinsicSurface`: `operatorMesh` → `intrinsic()`, the cotan Laplacian, masses, DEC, the vertex connection Laplacian (n-RoSy built from transport), Hodge–Dirac and `intrinsic_forms` rebuilt on it; `vertexFrames` / `vertexGrid` become geometry-central's (the `…Embedded` names kept as deprecated aliases); `geodesic.cpp` and `vector_heat.cpp` on the intrinsic surface; compute goldens that are gauge-free re-checked within tolerance, Studio-parity goldens replaced by section 4's tests | `compute/src`, `compute/include/nxr/compute.h`, `compute/test`; the compute bindings (`bindings/mex`, `bindings/node`, `bindings/wasm`, `bindings/python`) only where a name changes | G2 | as G1. Pipelines compile unchanged (the public compute API keeps its names) |
| G4 | the charts and frames on `IntrinsicSurface`: `ChartKit` a thin wrapper of `HeatMethodDistanceSolver` / `VectorHeatMethodSolver` (same API, the new gauge), `gaugeHeading` on `vertexTangentBasis`, `frames.h` as section 6, `ladder`, `quadrature`, `crossings`, the optical flow's `Corners` rebuilt; their tests and goldens | `compute/` only, plus expected values in the pipeline tests that read a gauge-dependent number | G3 | C4, C5, migration 6 to 8. **Overlap:** migration step 9 moves `pipelines/test/*` onto the task API; land G4 before it starts, or rebase G4 (the overlap is expected-value lines) |
| G5 | the gauge in the executor: `PortType::spin`, `PortType::gauge`, `check`'s refusal; `geometry.frame_rotate`, `geometry.frame_embed`, `geometry.frame_project` as nodes with their traits and tests | `pipelines/graph/` (types, check, a new node file) | **C5** (it moves the steps into `pipelines/graph/nodes/`; the type change is additive, but landing after C5 spares it a rebase) | migration 6 to 10 |
| G6a | the gauge record in rows and provenance; refusal of old-gauge data; `"gauge": "canonical frames"` → `"frame": "canonical"`; `<surface>_canonical_gauge_angle` recomputed: the small steps | `pipelines/src/frames.cpp`, `tiles.cpp`, `crossings_steps.cpp`, `pipelines/include/nxr/pipelines/frames.h`, `tiles.h` | G4, G5; **C4's `crossings_steps.cpp` PR and C5** | migration 6 to 10 |
| G6b | the same for the large files | `pipelines/src/meeg.cpp`, `pyramid.cpp`, `group.cpp`, `measure.cpp`, `eigen.cpp`, `spatial_band.cpp`, `patches.cpp`; `cli/meeg_command.cpp` | G6a; **migration 6 (`meeg.cpp`) and 7 (the command line)** | migration 8 to 10 |
| G7 | retire the port: delete `delaunay.h` / `delaunay.cpp`, the port inside `chart.cpp`, `compute/test/parity/delaunay_reference.mts` and `test_delaunay_parity.py`, `delaunay_dump.cpp`, the Studio-parity golden generators and goldens of section 8 item 6, the `…Embedded` aliases; `tools/geometry_inventory.py` reports 0 PORT uses (a test runs it); `docs/reference.md`, `compute/CONTRIBUTING.md` (its gauge and operator-mesh paragraphs) updated | `compute/`, `docs/`, bindings' deprecated names | G6b | migration 10 |
| G8 | the weekly upstream build (7), native and WebAssembly | `.github/workflows/` (a new workflow) | G1 | anything |

Main stays green at each step: G1, G2 and G8 only add; G3 and G4 change numbers only where the gauge or the normal enters, and replace each
affected parity golden with an analytic test in the same PR; G5 adds a check that every existing graph passes (no stored data has a spin yet);
G6a and G6b write the new records and refuse old ones in the same PR that recomputes the test datasets' fields.

## 10. Open questions, each with a recommendation

| # | question | options | recommendation |
|---|---|---|---|
| 1 | Which intrinsic triangulation | `SignpostIntrinsicTriangulation` · `IntegerCoordinatesIntrinsicTriangulation` | **signposts** (2.2): its tangent spaces are the input's through every flip; integer coordinates re-derive angles from a moving `v.halfedge()` |
| 2 | Default mass | `input-mixed-voronoi` (as now) · geometry-central's `barycentric` · `galerkin` | **keep `input-mixed-voronoi`** as the default through the `computeVertexDualAreas` override, and revisit only if section 4's measured errors favour another |
| 3 | The normal of the one tangent plane | geometry-central's angle-weighted `vertexNormals` · the port's area-weighted | **geometry-central's**: one plane for `vertexTangentBasis`, the canonical frames and every projection; the canonical frames are recomputed (G6a) |
| 4 | Boundary cone target | geometry-central's π · the port's (none) through an override | **geometry-central's** (3.4); cortical hemispheres have no boundary; the record counts boundary vertices |
| 5 | The 3D frame when ε > 0 | realise it from the mollified angles (2.3) · use `vertexTangentBasis` of the unmollified input and accept an O(ε) mismatch | **realise it from the mollified angles**, with the test that it equals `vertexTangentBasis` when ε = 0 |
| 6 | Old-gauge data | refuse with a message · convert automatically · recompute silently | **refuse with a message naming the run to repeat**; development data is recomputed in one pass; no automatic conversion (its rotation would depend on the retired port) |
| 7 | `GaugeType { Euclidean, LeviCivita, Trivial }` in `compute.h` uses "gauge" for the choice of connection | keep · rename to `ConnectionType` | **rename in G7** (`gauge()` facet → `connection()`, old names as deprecated aliases for one release), so "gauge" means one thing |
| 8 | The `[V,3,3]` canonical frames field | keep as a gauge-free export · derive on read only | **keep** while the Studio reads it (8, item 4); written from the angle array and `vertexGrid`, never a second source of truth |
| 9 | Granularity of the gauge id | library version only · version + surface + fingerprint | **version + surface + fingerprint** (5.1): the gauge depends on the face order of the surface and on ε, not only on the library |
| 10 | n-RoSy connection Laplacian | rebuilt from `transportVectorsAlongHalfedge^n` · only n = 1 | **rebuild** (a few lines, as geometry-central's `direction_fields.cpp` does), tested by A2's l(l + 1) − n² |
| 11 | Flip refusal "new length ≤ 0" | add it to an override · accept geometry-central's (non-finite only) | **accept**, with the test that every length is positive after the flips |
| 12 | The names `ChartKit`, `HeatMethod` | keep as wrappers · replace by the geometry-central solvers at call sites | **keep `ChartKit`** (a wrapper per connected piece, NaN off the piece, the `disc` validity rule) and **drop `HeatMethod`** (callers use `ChartKit` or the solver) |

## Checklist (the architecture policy)

- Each new compute function can be called with a matrix from anywhere: `IntrinsicSurface` takes positions and faces; `geometry.frame_rotate` takes arrays.
- No reduction moves into a node: every node here is pointwise.
- No file access, no io: `IntrinsicSurface` and the frame nodes are in `compute/` and `pipelines/graph/`; the gauge record is written by the runner
  through io's task API.
- Every choice of data stays the caller's; every method choice is a named, recorded option with its default stated.

## Appendix A: the inventory (`tools/geometry_inventory.py`)

Generated by `tools/geometry_inventory.py` on `main` at the date of this spec; rerun after every PR of section 9 (the PORT counts fall to zero
in G7). Comments are not counted. A symbol can be counted twice in a line when two patterns match (e.g. `delaunay::Mesh` and `Mesh::mass`); the
counts are for tracking, not accounting.

### Summary

- files using the port or its gauge (PORT): **74**, uses: **438**
  - compute test: 17
  - compute source: 15
  - compute header: 8
  - pipeline (nodes, steps): 8
  - pipeline test: 7
  - binding test: 5
  - port itself: 4
  - golden generator: 3
  - binding (node): 2
  - binding (wasm): 2
  - parity reference: 2
  - binding (mex): 1
- files using geometry-central's gauge today (GC-FRAME): **15**
- files using BOTH gauges: **7**: `bindings/mex/src/nxr_compute_mex.cpp`, `compute/include/nxr/compute.h`, `compute/src/connection_laplacian.cpp`, `compute/src/facets.cpp`, `compute/src/geometry_grid.cpp`, `compute/src/vertex_frames.cpp`, `compute/test/test_geometry_bundle.cpp`

### Symbols and their replacement

| symbol | family | uses | files | replaced by |
|---|---|---|---|---|
| `delaunay.h include` | PORT | 19 | 19 | geometrycentral/surface/signpost_intrinsic_triangulation.h (+ intrinsic_mollification.h) |
| `delaunay::Mesh` | PORT | 35 | 15 | SignpostIntrinsicTriangulation (intrinsicMesh, edgeLengths, signpostAngle) over the input ManifoldSurfaceMesh |
| `delaunay::DelaunayOptions` | PORT | 27 | 16 | nxr option struct kept (mollify -> mollifyIntrinsic relativeFactor; flipEps -> IntrinsicTriangulation::triangleTestEPS) |
| `delaunay::makeMesh` | PORT | 4 | 3 | makeManifoldSurfaceMeshAndGeometry / EdgeLengthGeometry |
| `delaunay::mollify` | PORT | 3 | 3 | mollifyIntrinsic(mesh, edgeLengths, relativeFactor) on the INPUT lengths, before the triangulation is built |
| `delaunay::addSignposts` | PORT | 6 | 4 | SignpostIntrinsicTriangulation constructor (signpostAngle) |
| `delaunay::flipToDelaunay` | PORT | 5 | 4 | IntrinsicTriangulation::flipToDelaunay() (flips counted through edgeFlipCallbackList) |
| `delaunay::prepare` | PORT | 19 | 11 | nxr IntrinsicSurface (new, thin): mollifyIntrinsic -> SignpostIntrinsicTriangulation -> mass recorded -> flipToDelaunay |
| `delaunay::cotanLaplacian` | PORT | 3 | 3 | IntrinsicGeometryInterface::cotanLaplacian (requireCotanLaplacian) |
| `delaunay::edgeCotanWeights` | PORT | 1 | 1 | edgeCotanWeights (requireEdgeCotanWeights) |
| `delaunay::faceAreas` | PORT | 1 | 1 | faceAreas (requireFaceAreas) |
| `delaunay::galerkinMass` | PORT | 4 | 4 | vertexGalerkinMassMatrix (requireVertexGalerkinMassMatrix) |
| `delaunay::barycentricMass` | PORT | 5 | 3 | vertexLumpedMassMatrix / vertexDualAreas (geometry-central's lumped mass IS barycentric) |
| `delaunay::mixedVoronoiMass` | PORT | 9 | 7 | NONE: rebuilt on geometry-central (cornerAngles, edgeLengths, faceAreas of the mollified input), served through an overridden computeVertexDualAreas |
| `Mesh::mass (recorded)` | PORT | 19 | 10 | NONE: the recorded input mass, served by the IntrinsicSurface's computeVertexDualAreas override |
| `delaunay::rescaledAngle` | PORT | 6 | 4 | halfedgeVectorsInVertex[he].arg() (signpost: signpostAngle / vertexAngleScaling) |
| `delaunay::reverseAngle` | PORT | 4 | 3 | halfedgeVectorsInVertex[he.twin()].arg() |
| `delaunay::connectionLaplacian` | PORT | 2 | 2 | vertexConnectionLaplacian (nSym = 1); nSym > 1 rebuilt from transportVectorsAlongHalfedge^n as direction_fields.cpp does |
| `delaunay::dec / Dec` | PORT | 4 | 2 | requireDECOperators: d0, d1, hodge1, hodge2 (hodge0 from the chosen mass) |
| `delaunay::nonDelaunayEdges` | PORT | 7 | 5 | IntrinsicTriangulation::isDelaunay(e) counted |
| `delaunay immediates` | PORT | 3 | 3 | EdgeLengthGeometry::faceArea / cornerAngle / halfedgeCotanWeight / edgeCotanWeight |
| `Manifold::operatorMesh` | PORT | 12 | 9 | Manifold::intrinsic() -> the IntrinsicSurface (its IntrinsicGeometryInterface) |
| `Manifold::operator* caches` | PORT | 15 | 2 | the IntrinsicSurface's require* quantities |
| `chart::HeatMethod` | PORT | 20 | 3 | HeatMethodDistanceSolver on the IntrinsicSurface (mass through computeVertexDualAreas) |
| `chart::ChartKit` | PORT | 48 | 17 | VectorHeatMethodSolver + HeatMethodDistanceSolver on the IntrinsicSurface, per connected piece |
| `ChartKit::transported` | PORT | 16 | 9 | VectorHeatMethodSolver::transportTangentVector |
| `ChartKit::logMap` | PORT | 16 | 9 | VectorHeatMethodSolver::computeLogMap (LogMapStrategy::VectorHeat) |
| `chart::tangentBasis / Basis` | PORT | 21 | 12 | vertexTangentBasis (+ vertexNormals) of the input VertexPositionGeometry |
| `gaugeHeading / headingVector` | PORT | 16 | 6 | atan2 on vertexTangentBasis (nxr, one line each) |
| `frames::gaugeAngle` | PORT | 12 | 8 | canonical frame as angle array in geometry-central's gauge (vertexTangentBasis) |
| `frames::angleFromNorth` | PORT | 3 | 3 | geometry.frame_rotate(field, to=canonical, spin=1) node |
| `frames::vertexNormals` | PORT | 8 | 6 | vertexNormals (angle-weighted) of the input VertexPositionGeometry |
| `ladder::leviCivita` | PORT | 15 | 10 | VectorHeatMethodSolver::transportTangentVector (unchanged maths, new gauge) |
| `geometry::vertexFrames` | PORT | 20 | 14 | vertexTangentBasis (one frame; vertexFramesEmbedded becomes vertexFrames) |
| `geometry::vertexGrid` | PORT | 8 | 6 | vertexTangentBasis packed e1 + i e2 (one grid; vertexGridEmbedded becomes vertexGrid) |
| `flow::connection::Corners` | PORT | 22 | 5 | cornerScaledAngles + halfedgeVectorsInVertex of the mollified input geometry (rebuilt; same gauge as the triangulation) |
| `vertexFramesEmbedded` | GC-FRAME | 6 | 5 | kept, renamed vertexFrames |
| `vertexGridEmbedded` | GC-FRAME | 7 | 6 | kept, renamed vertexGrid |
| `vertexTangentBasis` | GC-FRAME | 16 | 6 | kept (the one 3D realisation of the gauge) |
| `halfedgeVectorsInVertex` | GC-FRAME | 6 | 3 | kept; read from the IntrinsicSurface |
| `transportVectorsAlongHalfedge` | GC-FRAME | 8 | 4 | kept; read from the IntrinsicSurface |

### Files

| file | kind | family | symbols (uses) |
|---|---|---|---|
| `bindings/mex/src/nxr_compute_mex.cpp` | binding (mex) | GC-FRAME+PORT | delaunay::DelaunayOptions (1), geometry::vertexFrames (1), geometry::vertexGrid (1), halfedgeVectorsInVertex (1) |
| `bindings/node/index.mjs` | binding (node) | PORT | ChartKit::logMap (1), geometry::vertexFrames (1) |
| `bindings/node/src/addon.cpp` | binding (node) | PORT | delaunay::DelaunayOptions (1), geometry::vertexFrames (1) |
| `bindings/wasm/js/wrap.mjs` | binding (wasm) | PORT | ChartKit::logMap (2) |
| `bindings/wasm/src/nxr_compute_wasm.cpp` | binding (wasm) | PORT | geometry::vertexFrames (2) |
| `bindings/mex/test/explore_cortex.m` | binding test | PORT | ChartKit::logMap (1) |
| `bindings/node/test/test_intrinsic_delaunay.mjs` | binding test | PORT | geometry::vertexFrames (1) |
| `bindings/node/test/test_vertex_frames.mjs` | binding test | PORT | geometry::vertexFrames (2) |
| `bindings/scripts/_smoke-addon-errors.mjs` | binding test | PORT | ChartKit::logMap (1) |
| `bindings/scripts/_smoke-wasm.mjs` | binding test | PORT | ChartKit::logMap (2) |
| `compute/include/nxr/compute.h` | compute header | GC-FRAME+PORT | Manifold::operator* caches (3), Manifold::operatorMesh (1), delaunay.h include (1), delaunay::DelaunayOptions (3), delaunay::Mesh (2), geometry::vertexFrames (1), geometry::vertexGrid (1), halfedgeVectorsInVertex (1), vertexFramesEmbedded (1), vertexGridEmbedded (1) |
| `compute/include/nxr/frames.h` | compute header | PORT | chart::ChartKit (2), frames::angleFromNorth (1), frames::gaugeAngle (1) |
| `compute/include/nxr/intrinsic_forms.h` | compute header | PORT | delaunay.h include (1), delaunay::Mesh (1) |
| `compute/include/nxr/ladder.h` | compute header | PORT | chart::ChartKit (1), ladder::leviCivita (1) |
| `compute/include/nxr/opticalflow.h` | compute header | PORT | chart::tangentBasis / Basis (1) |
| `compute/include/nxr/opticalflow_connection.h` | compute header | PORT | delaunay.h include (1), delaunay::DelaunayOptions (2), delaunay::Mesh (3), flow::connection::Corners (7) |
| `compute/include/nxr/quadrature.h` | compute header | PORT | chart::ChartKit (1) |
| `compute/include/nxr/transport.h` | compute header | PORT | ladder::leviCivita (1) |
| `compute/src/connection_laplacian.cpp` | compute source | GC-FRAME+PORT | Manifold::operatorMesh (2), delaunay::Mesh (2), delaunay::connectionLaplacian (1), delaunay::rescaledAngle (1), delaunay::reverseAngle (1), geometry::vertexFrames (1), transportVectorsAlongHalfedge (2), vertexTangentBasis (3) |
| `compute/src/covariant_differential.cpp` | compute source | PORT | geometry::vertexFrames (5) |
| `compute/src/curvatures.cpp` | compute source | GC-FRAME | vertexTangentBasis (2) |
| `compute/src/dirac_operator.cpp` | compute source | GC-FRAME | vertexFramesEmbedded (1) |
| `compute/src/direction_field.cpp` | compute source | GC-FRAME | transportVectorsAlongHalfedge (2), vertexTangentBasis (2) |
| `compute/src/facets.cpp` | compute source | GC-FRAME+PORT | delaunay::cotanLaplacian (1), geometry::vertexGrid (1), vertexFramesEmbedded (1), vertexGridEmbedded (1) |
| `compute/src/frames.cpp` | compute source | PORT | chart::ChartKit (2), frames::angleFromNorth (1), frames::gaugeAngle (3), gaugeHeading / headingVector (1) |
| `compute/src/gauge.cpp` | compute source | GC-FRAME | transportVectorsAlongHalfedge (2) |
| `compute/src/geodesic.cpp` | compute source | PORT | Manifold::operatorMesh (1), chart::HeatMethod (2) |
| `compute/src/geometry_bundle.cpp` | compute source | GC-FRAME | halfedgeVectorsInVertex (4), transportVectorsAlongHalfedge (2), vertexGridEmbedded (1) |
| `compute/src/geometry_grid.cpp` | compute source | GC-FRAME+PORT | geometry::vertexFrames (1), geometry::vertexGrid (1), vertexFramesEmbedded (2), vertexGridEmbedded (1) |
| `compute/src/hodge_dirac.cpp` | compute source | PORT | Manifold::operatorMesh (1), delaunay::DelaunayOptions (1), delaunay::Mesh (4), delaunay::dec / Dec (2), delaunay::prepare (1) |
| `compute/src/intrinsic_forms.cpp` | compute source | PORT | Mesh::mass (recorded) (1), delaunay::Mesh (1), delaunay::prepare (1) |
| `compute/src/ladder.cpp` | compute source | PORT | ChartKit::transported (1), chart::ChartKit (1), ladder::leviCivita (1) |
| `compute/src/mesh_operators.cpp` | compute source | PORT | Manifold::operator* caches (12), Manifold::operatorMesh (3), Mesh::mass (recorded) (3), delaunay::DelaunayOptions (2), delaunay::Mesh (5), delaunay::barycentricMass (3), delaunay::cotanLaplacian (1), delaunay::dec / Dec (2), delaunay::galerkinMass (1), delaunay::prepare (1) |
| `compute/src/opticalflow.cpp` | compute source | PORT | chart::tangentBasis / Basis (2), delaunay.h include (1), delaunay::galerkinMass (1), delaunay::prepare (1), frames::vertexNormals (1) |
| `compute/src/opticalflow_connection.cpp` | compute source | PORT | Mesh::mass (recorded) (2), delaunay immediates (1), delaunay::DelaunayOptions (2), delaunay::Mesh (1), delaunay::addSignposts (1), delaunay::connectionLaplacian (1), delaunay::flipToDelaunay (1), delaunay::makeMesh (1), delaunay::mollify (1), delaunay::rescaledAngle (1), flow::connection::Corners (8) |
| `compute/src/quadrature.cpp` | compute source | PORT | ChartKit::transported (1), chart::ChartKit (1), gaugeHeading / headingVector (2) |
| `compute/src/smooth_field.cpp` | compute source | GC-FRAME | vertexTangentBasis (2) |
| `compute/src/transport.cpp` | compute source | PORT | ladder::leviCivita (1) |
| `compute/src/vector_heat.cpp` | compute source | GC-FRAME | vertexTangentBasis (4) |
| `compute/src/vertex_frames.cpp` | compute source | GC-FRAME+PORT | Manifold::operatorMesh (1), delaunay::Mesh (1), geometry::vertexFrames (1), vertexFramesEmbedded (1), vertexTangentBasis (3) |
| `compute/test/delaunay_dump.cpp` | compute test | PORT | Mesh::mass (recorded) (1), delaunay immediates (1), delaunay.h include (1), delaunay::DelaunayOptions (1), delaunay::Mesh (1), delaunay::nonDelaunayEdges (1), delaunay::prepare (1) |
| `compute/test/test_chart.cpp` | compute test | PORT | ChartKit::logMap (4), ChartKit::transported (6), chart::ChartKit (2), chart::tangentBasis / Basis (4), gaugeHeading / headingVector (7) |
| `compute/test/test_covariant_differential.cpp` | compute test | PORT | Manifold::operatorMesh (1), delaunay::edgeCotanWeights (1) |
| `compute/test/test_crossings.cpp` | compute test | PORT | chart::ChartKit (1), chart::tangentBasis / Basis (2), frames::gaugeAngle (1), ladder::leviCivita (1) |
| `compute/test/test_delaunay.cpp` | compute test | PORT | Manifold::operatorMesh (1), Mesh::mass (recorded) (3), delaunay::DelaunayOptions (1), delaunay::Mesh (7), delaunay::faceAreas (1), delaunay::nonDelaunayEdges (3), delaunay::prepare (6), geometry::vertexFrames (1) |
| `compute/test/test_facets.cpp` | compute test | GC-FRAME | vertexGridEmbedded (2) |
| `compute/test/test_frames.cpp` | compute test | PORT | chart::ChartKit (1), frames::angleFromNorth (1) |
| `compute/test/test_gauge_workflow.cpp` | compute test | PORT | geometry::vertexGrid (2) |
| `compute/test/test_geometry_bundle.cpp` | compute test | GC-FRAME+PORT | geometry::vertexFrames (1), geometry::vertexGrid (2), vertexGridEmbedded (1) |
| `compute/test/test_integrals.cpp` | compute test | PORT | delaunay::DelaunayOptions (1), delaunay::prepare (1) |
| `compute/test/test_intrinsic_forms.cpp` | compute test | PORT | delaunay.h include (1), delaunay::prepare (3) |
| `compute/test/test_ladder.cpp` | compute test | PORT | ChartKit::transported (1), chart::ChartKit (2), frames::gaugeAngle (2), frames::vertexNormals (1), ladder::leviCivita (2) |
| `compute/test/test_monogenic.cpp` | compute test | PORT | delaunay.h include (1), delaunay::mixedVoronoiMass (1), delaunay::prepare (1) |
| `compute/test/test_opticalflow.cpp` | compute test | PORT | chart::tangentBasis / Basis (1) |
| `compute/test/test_opticalflow_connection.cpp` | compute test | PORT | flow::connection::Corners (3) |
| `compute/test/test_quadrature.cpp` | compute test | PORT | chart::ChartKit (2), chart::tangentBasis / Basis (2), gaugeHeading / headingVector (2) |
| `compute/test/test_transport.cpp` | compute test | PORT | ladder::leviCivita (4) |
| `compute/test/test_vector_covariant_operators.cpp` | compute test | PORT | geometry::vertexFrames (1) |
| `compute/test/golden/gen_chart_golden.mts` | golden generator | PORT | ChartKit::logMap (1), ChartKit::transported (2) |
| `compute/test/golden/gen_ladder_golden.mts` | golden generator | PORT | ChartKit::transported (1) |
| `compute/test/golden/gen_opticalflow_connection_golden.mts` | golden generator | PORT | flow::connection::Corners (2) |
| `compute/test/parity/delaunay_reference.mts` | parity reference | PORT | Mesh::mass (recorded) (1), delaunay::addSignposts (1), delaunay::flipToDelaunay (1), delaunay::mollify (1), delaunay::nonDelaunayEdges (1) |
| `compute/test/parity/test_delaunay_parity.py` | parity reference | PORT | Mesh::mass (recorded) (1) |
| `pipelines/src/crossings_steps.cpp` | pipeline (nodes, steps) | PORT | chart::ChartKit (1), chart::tangentBasis / Basis (2), frames::gaugeAngle (1), frames::vertexNormals (2), ladder::leviCivita (1) |
| `pipelines/src/eigen.cpp` | pipeline (nodes, steps) | PORT | Manifold::operatorMesh (1), delaunay::DelaunayOptions (1), delaunay::Mesh (1) |
| `pipelines/src/frames.cpp` | pipeline (nodes, steps) | PORT | chart::ChartKit (2), chart::tangentBasis / Basis (1), frames::gaugeAngle (1), frames::vertexNormals (1) |
| `pipelines/src/measure.cpp` | pipeline (nodes, steps) | PORT | Mesh::mass (recorded) (2), delaunay.h include (1), delaunay::mixedVoronoiMass (1) |
| `pipelines/src/meeg.cpp` | pipeline (nodes, steps) | PORT | delaunay.h include (1), delaunay::DelaunayOptions (2), delaunay::Mesh (2), delaunay::prepare (2) |
| `pipelines/src/patches.cpp` | pipeline (nodes, steps) | PORT | delaunay.h include (1) |
| `pipelines/src/spatial_band.cpp` | pipeline (nodes, steps) | PORT | delaunay.h include (1), delaunay::mixedVoronoiMass (1) |
| `pipelines/src/tiles.cpp` | pipeline (nodes, steps) | PORT | ChartKit::transported (1), chart::ChartKit (4), chart::tangentBasis / Basis (1), frames::gaugeAngle (2), frames::vertexNormals (1), ladder::leviCivita (2) |
| `pipelines/test/test_hodge_power.cpp` | pipeline test | PORT | delaunay.h include (1), delaunay::prepare (1) |
| `pipelines/test/test_measure_pipeline.cpp` | pipeline test | PORT | delaunay.h include (1), delaunay::mixedVoronoiMass (1) |
| `pipelines/test/test_opticalflow_pipeline.cpp` | pipeline test | PORT | delaunay.h include (1) |
| `pipelines/test/test_spatial_band.cpp` | pipeline test | PORT | delaunay.h include (1) |
| `pipelines/test/test_spatial_hat.cpp` | pipeline test | PORT | delaunay.h include (1) |
| `pipelines/test/test_speedspectrum_pipeline.cpp` | pipeline test | PORT | delaunay.h include (1) |
| `pipelines/test/test_vortex_cores_pipeline.cpp` | pipeline test | PORT | chart::ChartKit (1), chart::tangentBasis / Basis (2), frames::gaugeAngle (1), frames::vertexNormals (2), ladder::leviCivita (1) |
| `compute/include/nxr/chart.h` | port itself | PORT | ChartKit::logMap (2), ChartKit::transported (1), chart::ChartKit (7), chart::HeatMethod (7), chart::tangentBasis / Basis (1), delaunay.h include (1), delaunay::DelaunayOptions (1), delaunay::Mesh (2), gaugeHeading / headingVector (2) |
| `compute/include/nxr/delaunay.h` | port itself | PORT | delaunay::DelaunayOptions (4), delaunay::addSignposts (1), delaunay::barycentricMass (1), delaunay::flipToDelaunay (1), delaunay::galerkinMass (1), delaunay::makeMesh (1), delaunay::mixedVoronoiMass (1), delaunay::nonDelaunayEdges (1), delaunay::rescaledAngle (1), delaunay::reverseAngle (1) |
| `compute/src/chart.cpp` | port itself | PORT | ChartKit::logMap (2), ChartKit::transported (2), Mesh::mass (recorded) (2), chart::ChartKit (17), chart::HeatMethod (11), chart::tangentBasis / Basis (2), delaunay immediates (1), delaunay::DelaunayOptions (1), delaunay::Mesh (2), delaunay::cotanLaplacian (1), delaunay::mixedVoronoiMass (1), flow::connection::Corners (2), gaugeHeading / headingVector (2) |
| `compute/src/delaunay.cpp` | port itself | PORT | Mesh::mass (recorded) (3), delaunay.h include (1), delaunay::DelaunayOptions (3), delaunay::addSignposts (3), delaunay::barycentricMass (1), delaunay::flipToDelaunay (2), delaunay::galerkinMass (1), delaunay::makeMesh (2), delaunay::mixedVoronoiMass (3), delaunay::mollify (1), delaunay::nonDelaunayEdges (1), delaunay::rescaledAngle (3), delaunay::reverseAngle (2) |

### Goldens read by the affected tests

| golden | read by | generator |
|---|---|---|
| `chart_golden.txt` | `compute/test/test_chart.cpp` | gen_chart_golden.mts |
| `crossings_golden.txt` | `compute/test/test_crossings.cpp` | gen_crossings_golden.py |
| `repeat_golden.txt` | `compute/test/test_crossings.cpp` | gen_repeat_golden.py |
| `frames_golden.txt` | `compute/test/test_frames.cpp` | gen_frames_golden.m |
| `integrals_golden.txt` | `compute/test/test_integrals.cpp` | gen_integrals_golden.py |
| `ladder_golden.txt` | `compute/test/test_ladder.cpp` | gen_ladder_golden.mts |
| `hodge_weights_golden.txt` | `compute/test/test_opticalflow.cpp` | gen_hodge_weights_golden.py |
| `opticalflow_golden.txt` | `compute/test/test_opticalflow.cpp` | gen_opticalflow_bst_golden.m, gen_opticalflow_connection_golden.mts, gen_opticalflow_golden.m |
| `opticalflow_connection_golden.txt` | `compute/test/test_opticalflow_connection.cpp` | gen_opticalflow_connection_golden.mts |
| `quadrature_golden.txt` | `compute/test/test_quadrature.cpp` | gen_quadrature_golden.py |

### Stored names (string literals in pipelines/, io/, cli/)

| pattern | what it is, and what the switch does | files |
|---|---|---|
| `_frame_singularities\b` | Selection of singular vertices: gauge-free; kept | `pipelines/src/frames.cpp` |
| `"gauge"` | a row/provenance that already names a gauge | `cli/meeg_command.cpp`, `pipelines/src/crossings_steps.cpp`, `pipelines/src/group.cpp`, `pipelines/src/measure.cpp`, `pipelines/src/meeg.cpp`, `pipelines/src/pyramid.cpp`, `pipelines/src/tiles.cpp` |
| `_anatomical_(angle|second|handed|reliable)\b|_anatomical_` | anatomical frame: angle FROM NORTH (gauge-free); becomes a reporting projection | `pipelines/src/frames.cpp` |
| `_canonical_frames\b` | canonical frames Field [V,3,3] (e1 north, e2 west, n): 3D vectors, gauge-free; kept | `io/src/catalogue.cpp`, `pipelines/src/frames.cpp`, `pipelines/src/group.cpp`, `pipelines/src/meeg.cpp`, `pipelines/src/pyramid.cpp`, `pipelines/src/tiles.cpp` |
| `eigenmodes\b|_eigenvectors\b` | eigenmodes (scalar: gauge-free; complex connection modes: gauge recorded) | `io/schema/nxr.manifold.schema.json`, `io/schema/nxr.maps.schema.json`, `io/schema/nxr.subject.schema.json`, `io/schema/seed/manifold.schema.json`, `pipelines/src/bound.cpp`, `pipelines/src/cellquadrature.cpp`, `pipelines/src/eigen.cpp`, `pipelines/src/meeg.cpp`, `pipelines/src/patches.cpp`, `pipelines/src/spatial_band.cpp` |
| `_canonical_gauge_angle\b` | north's angle in the PORT's signpost gauge: recomputed in geometry-central's gauge (gauge recorded) | `pipelines/src/frames.cpp` |
| `flow_velocity(_anterior|_second)?\b|velocity_(anterior|second)\b` | optical-flow velocity: embedded 3D (gauge-free) + anatomical components | `io/schema/measures_nxr.json`, `pipelines/src/meeg.cpp` |
| `levi-civita` | tiles' levi-civita form (ChartKit::transported in the port's gauge): recomputed, gauge recorded | `cli/meeg_command.cpp`, `pipelines/include/nxr/pipelines/tiles.h`, `pipelines/src/tiles.cpp` |

## Appendix B: the comparison with the old port

Filled by G2.

## Appendix B: the one-time comparison with the port (G2, `compute/test/test_gc_port_comparison.cpp`)

Synthetic meshes (an icosphere, a jittered icosphere, a UV sphere with poles, a torus, a jittered planar disc), the port and `IntrinsicSurface` built on each. `flipEps` is 1e-12
in the comparison (geometry-central flips at `<= -eps`, so an exact 0 would flip cocircular ties for ever; the port flips below `-eps`).

| item | result |
|---|---|
| 1 gauge rotation | `K_gc = D K_port D^H`, D = diag(e^{iθ_v}), θ_v the angle of the port's reference halfedge in geometry-central's gauge (the other sign gives O(1) differences): max difference over interior entries 1.8e-15 (icosphere), 2.6e-13 (jittered), 4.4e-15 (UV), 6.4e-15 (torus), 2.7e-15 (disc) |
| 2 triangulation | the same number of flips on every mesh (0, 102, 0, 0, 8); cotan Laplacians equal to 1.6e-13; at the default `flipEps` 1e-6 also equal (no near-tie edges on these meshes) |
| 3 boundary | 30 boundary vertices (disc) in both; entries touching a boundary vertex differ by up to 0.15: the π rescaling (section 3.4), as documented |
| 4 masses | mixed Voronoi, barycentric and Galerkin equal to 1e-15 on every mesh |
| 5 frames | angle between the port's e1 and geometry-central's e1 after removing θ_v: median 1e-3 to 1e-2 rad; the tail (99%) 0.002 to 0.03 rad on the regular meshes, 1.7 rad at a few vertices of the strongly jittered sphere (cone angle far from 2π: the average of the edges rotated back differs from the reference edge there); it is the 3D realisation only |
| 6 charts | heat distance equal to 1.6e-5 (relative 5e-6) on the closed meshes; on the disc the distances differ (the port takes the Dirichlet average at a boundary, geometry-central does not); log map angle after the rotation by θ_source: median 0.009 to 0.036 rad on the regular meshes (the two log maps are different estimators), 0.34 rad on the jittered sphere (not a chart there) |

**Poor meshes** (same file; a UV sphere with every ring jittered in longitude and latitude, so skewed needle and obtuse triangles on the unit sphere), intrinsic against embedded:

| mesh | flips | eigenvalue error (l ≤ 3) | heat distance error from a pole | negative cotan weights (embedded) |
|---|---|---|---|---|
| needle 6×96, skew 0.45 | 1126 | 0.133 / 0.112 | **0.037 / 0.43** | 0% / 31%, worst −3.1 |
| needle 8×128, skew 0.45 | 2286 | 0.083 / 0.107 | **0.032 / 0.87** | 0% / 32%, worst −3.8 |
| jittered icosphere 3 (0.7) | 778 | 0.33 / 0.30 | 0.22 / 0.13 | 0% / 24%, worst −5.5 |

Correction to an expectation: the cotangent Laplacian is the P1 stiffness matrix, positive semidefinite for any valid triangulation, so the embedded one is positive semidefinite too (smallest
eigenvalue 1e-13 to 1e-12 in every case). What obtuse triangles break is the **discrete maximum principle** (negative off-diagonal weights, 24 to 32% of the edges here), and with it the heat
method's distance (10 to 25 times worse on the needle meshes). The intrinsic Delaunay triangulation removes every negative weight; eigenvalue errors are of the same order for both, because the
spectrum error is dominated by the mesh's own geometric error. The jittered sphere (large jitter, little needle shape) is where intrinsic does not help the distance either.

## Appendix C: what G4 did (charts and frames on `IntrinsicSurface`)

- `ChartKit` (`chart.h`, `chart.cpp`) wraps `HeatMethodDistanceSolver` and `VectorHeatMethodSolver` on one `IntrinsicSurface` per connected piece, the same API; `chart::HeatMethod` and the port's `tangentBasis` are gone from the library, and
  `Basis` is double (the float32 rounding is dropped). `gaugeHeading` / `headingVector` read the gauge's 3D basis (`IntrinsicSurface::tangentBasis`). The kit equals the solvers called directly to 1e-12; the geodesic default
  (`solve::heat`) equals the kit's distance exactly (G3 had 1e-4 against the port). **Log map:** `theta` is the angle of `computeLogMap` (`LogMapStrategy::VectorHeat`); `r` stays the heat distance, because the log map's own
  radius is the coarser estimate (2.6% against the sphere's arccos on the 642-vertex icosphere for the heat distance, 8% for |z|).
- The Studio's chart golden and its generator (`chart_golden.txt`, `gen_chart_golden.mts`) are deleted: `test_chart` checks the wrapper against geometry-central and the sphere / plane analytically instead.
- The port's chart code is kept for the one-time comparison only, in `compute/test/chart_port.*` (namespace `nxr::chart_port`); G7 deletes it with the rest of the port.
- `frames.h` as section 6: `gaugeAngle(IntrinsicSurface&, direction)` (the canonical frame's stored form), `vertexNormals(IntrinsicSurface&)` (geometry-central's, and the array form of the same formula),
  `angleFromNorth` removed (a pipeline step: `geometry.frame_rotate`, then `arg`), and the anatomical axes as `anatomicalProjection`: frame-free for the SCS definition (3D axes, `anatomicalComponents(a, vectors)`, no gauge). Kept until
  the pipelines move (G6a), because the pipelines compile unchanged in G4: `gaugeAngle(ChartKit&, …)`, `anatomicalFrame`, `Anatomical::reference`.
- `ladder`, `quadrature` and `crossings` are unchanged code on the new kit. The Levi-Civita rho is gauge-free; against the Studio's transport it now agrees to the two gauges' 3D realisations (sphere 1e-3, irregular mesh a median of
  6e-3 rad), and the exact check is analytic: parallel transport along the great circle on the unit sphere, 1.8e-3 rad on the 2562-vertex icosphere (`test_ladder`).
- Not done in G4: the optical flow's `Corners` and `componentPencil` (`opticalflow_connection.cpp`) still use the port; they carry the Studio's float32 parity and their own golden, and leave with the port.
