# One coordinate system on the cortex for flow analysis: frames (gauge), charts (log map), tiles (dyadic)

2026-11-11. Read-only audit, requested to find out how the Studio implements a coordinate system on the cortex.

Scope: the Cortical Flow Studio (`nxr-cortical-flow/studio/src`, TypeScript), nxr-neuro `compute/` (C++, geometry-central), nxr
(`src/nxr`, Python), Rheome (`+rheome`, MATLAB). Paths below are relative to each repo. Nothing was run.

**Why it matters (the gap).** A flow is a field of tangent vectors. To report a direction ("this wave travels anterior"), to average
directions over a tile or across subjects, or to read a phase gradient in a window, every vector must be written in a frame whose
meaning is known (a **gauge**), moved between vertices by a known **connection**, read in a local **chart** that is valid where it is
used, and summarised on **tiles** whose scale is declared. Today the four codebases each have some of these pieces, built in different
ways. nxr-neuro has the most operators and the fewest of these choices fixed.

---

## 1. The Studio

### 1.1 The mesh every intrinsic quantity lives on

- `toolbox/differential/intrinsicDelaunay.ts`: `mollify` (:159; δ = 1e-6 × mean edge, geometry-central's `relativeFactor`) →
  `addSignposts` (:126) → `flipToDelaunay` (:232). The lumped mass (:280) is the **mixed-Voronoi area of the input triangulation**,
  recorded before the first flip.
- **Signposts** (:121-135): around each vertex, starting at its **first outgoing halfedge in face order** (on a boundary, the
  twinless one), counter-clockwise, the angle grows by each corner. `ref[v]` = head of that first halfedge, an input edge.
- Rescaled angle (`connectionLaplacian.ts:23`): ψ_h = coneTarget(v) · angle_h / angleSum_v, with coneTarget = 2π inside and the
  measured angle sum on the boundary (:31).

### 1.2 The tangent frame per vertex ("C's gauge", the signpost gauge)

`connectionLaplacian.ts:80` `tangentBasis` (geometry-central's `vertexTangentBasis`, but with the signpost reference):
- n = area-weighted normal of the input faces;
- e1 = the reference input edge (v → ref[v]) projected on the tangent plane (ψ = 0);
- e2 = n × e1 (ψ = π/2).
A complex value z = a + ib at v means a e1 + b e2 (`embedTangent`, :104).

This frame is **arbitrary from vertex to vertex**: it follows the mesh's halfedge order. Rheome measured it (`+operators/gauge.m:8-20`):
neighbouring half-edge frames differ by **79°** on average (vector-diffusion and trivial-connection frames: 20-22°).

### 1.3 The connection (parallel transport)

`connectionLaplacian.ts:1-17, :46` (port of geometry-central `vertexConnectionLaplacian` / `transportVectorsAlongHalfedge`; Sharp,
Soliman and Crane 2019, §5):

    (L u)_i = Σ_j w_ij (u_i − e^{i r_ji} u_j),   r_ji = ψ_ij + π − ψ_ji   (raised to nSym for n-RoSy)

w_ij are the cotan weights on the intrinsic Delaunay mesh. The operator is stored as a real 2n × 2n matrix, and its mass is the lumped
mass twice. The connection is **Levi-Civita**, and the gauge is the signpost frame. The face-wise crossing between an embedded vector
and C's gauge is `intrinsicGauge.ts`:
- `cornersOf` (:32);
- `intrinsicGradient` (:63), a scalar gradient in C's gauge;
- `embedAt` (:97), C → embedded;
- `toIntrinsic` (:121), embedded → C. Exact where a preimage exists; it has gaps at folded fans (97.7 % within 1e-2 rad on the cortex
  fixture).

### 1.4 The log map (local polar chart)

`compute/runtime/surfaceKit.ts:802-823` `logMap(v)` gives **geodesic polar coordinates**:
- **r**: the heat-method distance (`HeatMethod`, `toolbox/differential/heatMethod.ts:6, :32-40`). It solves (M + tL)u = δ, then the
  Poisson step, with **t = (mean intrinsic edge)²** (tCoef = 1), on the **intrinsic Delaunay** mesh.
- **θ**: the angle of the distance's vertex gradient (the radial direction) relative to the seed's reference vector carried there by
  the **vector heat method**: θ = arg(∇r / X). The transport is `transported` (:577):

      (C + M/t) X = M δ_i e^{iψ},   t = mean edge²   (prefactored LDLᵀ, nested dissection; :581-590)

- θ ∈ (−π, π]. θ = 0 lies **along the seed's reference edge** (the signpost gauge). θ is NaN off the seed's component.
- Validity: a chart only inside the injectivity radius. On the pial mesh that is **inj(a) ≈ 10-11 mm** (median), about 0.4 % of a
  hemisphere (`docs/research/spacetime-wavelets/RQ-B-chart-signal-questions.md:11-15, :50`). The chart tools refuse an arc past the
  cut locus.
- Consumers:
  - Point and circle keys stored as (Distance, Bearing) in the origin's log map (`studio/tools/useGeoline.ts`, `computeCircle`);
  - atom centre paths (`surfaceKit.ts:1156-1200`);
  - the vortex finding of 2026-10-20 (`docs/research/spacetime-wavelets/FINDING-2026-10-20-alpha-vortex.md:12`).
- Other uses of the transport: the `transported` tile orientation (:892, the seed direction carried to every tile) and
  `headingOffsets` (:676). `smoothest` (:592) is the Knöppel smoothest field (inverse iteration on (C, M)).

### 1.5 The gauge that means the same in every subject: the canonical frames

The Studio does not compute this frame. It reads it from the subject (spec `docs/superpowers/specs/2026-10-05-canonical-gauge-design.md`):
- `<surface>_canonical_frames`: a `matrix3` Field [V,3,3] with rows **e1 = north, e2 = n × e1 = "west", n outward**. e1/e2 are NaN at
  singular vertices, which are also listed in `<surface>_frame_singularities` (adapter `adapters/nxr/index.ts:58-77, :115`; port
  `studio/ports.ts:33-43`).
- How it is made: solved on the subject's registered sphere (poles at ±z), then pushed to the cortex by 1-ring Jacobians. "North"
  agrees across subjects to a median of 1.4-1.7°, p90 11-13°.
- Used for:
  - the **atom heading** (0° = north on every subject). It is converted into the CL basis by `headingOf` (`surfaceKit.ts:537`) and
    `gaugeDirection` (:376-378); time atoms read it at each centre (spec Part 1 §4);
  - the scopes' **heading** readout: "the tile-mean tangential vector's angle in the TILE's frame (its mean normal, its mean e1 made
    tangent)", gauge e1 when present (`studio/scopes/cells.ts:13-15, :59-61`);
  - **carrying a vector to the template** (`toolbox/template/toTemplate.ts:67`): components (v·e1, v·e2) in the subject's canonical
    frame, averaged through the transpose of `_to_default`, rebuilt on the default's frame. Gauge-free quantities (|v|, div, curl)
    are computed on the subject and carried as scalars (spec Part 2).
- Fallback when no gauge is supplied: `kit.frames()` (`surfaceKit.ts:1456`), an **extrinsic** frame (x̂ projected, or ŷ where
  |n·x̂| ≥ 0.9; e2 = n × e1). That makes a **third** per-vertex frame in the Studio.

### 1.6 The dyadic tiles

`engine/ladder.ts` (copied from `backend/src/db/surface-ladder.ts`):
- **Pairing with scale by Weyl's law**: level ℓ has 2^ℓ equal-area tiles, paired with λ(m) = 4π m / A, wavelength 2π/√λ (:4-24, :57).
  The tower's level j is 2^j tiles **and** the Meyer frame's band at j, with λ band [2π·2^{j+1}/A, 4π·2^j/A] (`surfaceKit.ts:648-668`).
- **Construction** (`surfaceLadder`, :230):
  - each tile is halved by **equal mass across its geodesic diameter**;
  - the geodesic is **Dijkstra on the input edge graph** (Euclidean edge lengths), not the heat method: two sweeps (first vertex → a,
    a → b), a stable sort by d_a − d_b, cut at half the mass;
  - `onePiece` (:148) keeps each child connected;
  - mass is barycentric (A/3).
- Levels are arithmetic: tile at ℓ = code >> (D − ℓ), children 2t and 2t+1. It stops before a level would leave an empty tile.
- Kit API:
  - `labels(level)` (:626), unique across components;
  - `tile(level, v)` (:1517);
  - `tileCentres(level)` (:1525): the vertex nearest the tile's mass centroid, Euclidean, with the tile's mass;
  - `levels()` and `tower()` (:651, :659).
- **Three ladder implementations exist in the Studio repo**:
  - `engine/ladder.ts`;
  - the nxr-analysis port in `toolbox/measure/curlCoreTiles.ts:1-22` (bit-compatible, code = 2^ℓ·h + t);
  - `viewer/tiles.ts`, which uses `engine/ladder.ts` to average arrows per tile (area-weighted **ambient** 3D mean, :65) for print.
- **Vectors in tiles**:
  - Cells `heading`: the tile-mean ambient vector's angle in the tile-mean frame.
  - Viewer arrows: the ambient mean.
  - Curl-core tile crossings: unit **ambient** displacements projected on the local normal and summed per tile
    (`curlCoreTiles.ts:16-18`).
  - None of these transports along the connection.

### 1.7 Optical flow in the Studio

`toolbox/flow/opticalFlow.ts:1-30, :173`: manifold Horn–Schunck, with **smoothness vᵀCv using the connection Laplacian** (gauge-free).
This is a deliberate, ruled deviation (2026-10-04) from the MATLAB/Rheome a'Ka + b'Kb, which "depends on the gauge … with per-vertex
random frames the sphere oracle fell to cos 0.15-0.40". Other settings:
- it runs on the intrinsic Delaunay mesh, with lumped mass;
- vectors cross into and out of C's gauge face by face (`intrinsicGauge`);
- default α = 0.01;
- the output is embedded 3n per frame.

---

## 2. nxr-neuro `compute/`

| What | Where | Mesh | Frame / convention |
|---|---|---|---|
| Operator mesh: mollify → signposts → Delaunay flips, mixed-Voronoi mass | `include/nxr/delaunay.h:8-12, :62 (ref), :88-106` | intrinsic (default since, `compute.h:97-105`) | signposts as the Studio (first outgoing halfedge in face order) |
| Connection Laplacian | `delaunay.h:123` `connectionLaplacian(Mesh, nSym)`; `compute.h:735` `assembleConnectionLaplacian` (vertex/face/edge, Real2N), `:781` trivial-connection variant, `:748` connection gradient d^∇ | operator mesh | r_ji = ψ_ij + π − ψ_ji, as the Studio (`test_cf_parity.cpp`) |
| Per-vertex frames | `compute.h:1756-1772` `vertexFrames` = **the Studio's `tangentBasis`** (signpost ref); `vertexFramesEmbedded` = geometry-central's `vertexTangentBasis` (gc's own reference halfedge); `:1781-1783` `vertexGrid`, `vertexGridEmbedded`, `faceGrid` (c = e1 + i e2); `:1754` `frames` (face: e1 along f.halfedge()) | input / operator | two different vertex references; per-face frames as well |
| Gauges | `compute.h:83` `GaugeType {Euclidean, LeviCivita, Trivial}`; `Manifold::setGauge`, `gauge()` (`:163-178`); `facets.h:85-103` `GaugeFacet::grid()` = combed frame; `compute.h:1509` `computeTrivialConnection` (A β = −K + 2πσ), `:1551` `integrateTrivialGaugeRotations` (BFS from vertex 0 / face 0, one component), `:1567` `trivial` (direction field with prescribed singularities), `:1588` `smoothVertex` (Knöppel n-RoSy) | — | the trivial gauge is anchored at vertex 0, so its global rotation is **arbitrary**; nothing ties it to north |
| Vector heat transport | `compute.h:1355` `VectorHeatSolver(m, tCoef=1)`, `:1382` `parallel(solver, sources, vectors[nS×3])` → [V×3] world, `:1394` `extendScalar` | **`m.geometry()` = the INPUT mesh** (`src/vector_heat.cpp:28-30`), not the operator mesh | gc `vertexTangentBasis` (= `vertexFramesEmbedded`) |
| Log map | `compute.h:1420-1436` `logMap(solver, v, strategy=AffineLocal)` → `logCoords` [V×2] (x, y), plus source e1/e2 | **input mesh** | gc `computeLogMap`; default **Affine Heat Method** (`AffineLocal`), `VectorHeat`, `AffineAdaptive`; axes = gc's `vertexTangentBasis` at the source |
| Karcher mean | `compute.h:1441` `findCenter(solver, sources, p=2)` | input | — |
| Geodesic distance | `src/geodesic.cpp:16-27` gc `HeatMethodDistanceSolver(c.geometry())` | **input mesh** | — |
| Optical flow | `opticalflow.h:1-27, :46` `tangentBasis(normals)` = Rheome/Brainstorm **(1,1,1)×n** (`[n_z−n_y, n_x−n_z, n_y−n_x]`), smoothness **a'Ka + b'Kb** with the **embedded** cotan K; `opticalflow_methods.h:5-17` methods direct/pcg/modal (energies frame/diagonal/full) | embedded | gauge-dependent smoothness |
| Phase gradient | `quadrature.h:123-133` `weightedPhaseGradient(z, ops, faces, areas, window, e1, e2)`: Σ_f A_f P_f summed as **ambient 3D vectors**, then projected on one caller-given (e1, e2) | embedded | no chart, no transport; the window and its axes come from the caller |
| Surface ladder / tiles | **absent** (the "ladder" hits in `compute.h:837-995` are the eigensolver's shift ladder) | — | — |
| Canonical frames, sphere registration, anatomical axes | **absent** (no reader for `<surface>_canonical_frames` in `io/` or `pipelines/`) | — | — |

## 3. nxr (Python) and Rheome (MATLAB)

### nxr

- `geom/logmap.py:1-77` `GeometryKit`: a **line port of the Studio's** `transported` and `logMap`, per connected component. It uses:
  - intrinsic Delaunay (`operators.intrinsic_delaunay.prepare`, mollify 1e-6, flip_eps 1e-6);
  - `connection_laplacian` and `tangent_basis` (the signpost gauge);
  - the heat method, with t = mean edge² and splu of (C + M/t);
  - `log_map(v)` → (r, θ), θ = arg(∇r · conj(X)), θ(v) = 0;
  - `gauge_heading(v, d)`, the angle of a 3D direction in v's basis, so that **θ_gauge = θ − gauge_heading(v, north_v)**. This is how
    a bearing is read from canonical north.
- `geom/frames.py:1-41, :175-253`: the **canonical frames** builder.
  - `optimal_field` (:175): the trivial connection with +1 at both poles, then one global rotation fixed so the field aligns with −∇r
    of the log map from the north pole.
  - `hemisphere_frames(domain="sphere")` (:219): solved on the registered sphere, pushed forward by `jacobians` (:152; a 2×2
    least-squares fit on 1-ring edges). Vertices are singular if the condition number exceeds 50, plus the poles.
  - Poles: the vertices nearest fsaverage's ±z (`poles_on_sphere`, :278). On fsaverage +z is precentral and −z is fusiform / inferior
    temporal.
  - It runs through **`compute.mjs`**, the old **nxr-compute Node binding** (`frames.py:54-97`: `m.interpolate.trivial`,
    `m.uv.logMap`; `compute.mjs:36-53`). That is a dependency on the library nxr-neuro replaces.
- `geom/coords.py:1-36`, the **template coordinate system**:
  - positions as canonical latitude and longitude on the registered sphere (longitude 0 through the occipital pole);
  - `two_pole_coords`: latitude = 90° − 180°·d_N/(d_N + d_S); longitude = log-map bearing from the north pole;
  - polar charts by log map about anatomical landmarks (occipital and frontal poles, centroids of pericalcarine, precuneus,
    precentral, superior temporal and temporal pole);
  - bearings measured from canonical north.
  - Naming conflict: coords.py calls **n × e_north "east"** (:4). frames.py, the spec and the Studio call the same vector **"west"**.
    For a viewer outside the surface, with north up, n × north points left, which is west.
- `geom/ladder.py:1-30, :146`: `surface_ladder`, **bit-exact to the Studio's** (pinned by `tests/test_atlas_ladder.py`, including V8's
  sort and `Math.hypot`).
- `geom/trees.py:1-18`:
  - the SUBJECT tree is the ladder on the subject's pial;
  - the GROUP tree is built once on fsaverage, and each subject vertex takes the code of its nearest fsaverage vertex on the registered
    sphere;
  - codes are h·2^D + t, and roll-up is a shift.
- `measure/vectors.py:1-23`, the vector reductions over tiles:
  - **Closed form** (`vector_stats`, :66): area-weighted sums of c1 = v·e1, c2 = v·e2, cn, and second moments, in the canonical
    frames. The docstring says they "roll up like scalars", because e1 is parallel for the trivial connection, which is flat away from
    the poles.
  - **Levi-Civita** (`levi_civita` :212, `lc_resultant` :251):
    - per level, node centres (`node_centres`, :199; the vertex nearest the area-weighted centroid);
    - vector-heat transport of each centre's e1 gives ρ_ℓ(i);
    - the resultant is Σ a (c1 + i c2) e^{−iρ};
    - the inter-centre connection is Ω_ℓ[a, b].
    - It also goes through the old binding (`run_compute("transport")`).
- `detect/vortex_cores.py:62-68`: tile crossings as **ambient unit vectors summed** (projected on the local normal), with no frame.
- `opticalflow/scalar.py:1-44`: Rheome's a'Ka + b'Kb with the (1,1,1)×n basis, the **same as nxr-neuro, unlike the Studio**.
- `spectral/lagged.py:142-155` `phase_gradient_flux`: the Python twin of `weightedPhaseGradient`. Axes (b2, b1); window and chart from
  the caller.

### Rheome

- `+dynamics/tangent_basis.m:14-23`: Brainstorm's basis, e1 ∝ (n_z−n_y, n_x−n_z, n_y−n_x), with a fallback near n ∥ (1,1,1).
  - This is an **extrinsic combing**, singular where n ∥ ±(1,1,1)/√3.
  - It flips across folds.
  - It is the frame nxr-neuro's optical flow uses.
- `+operators/gauge.m:2-20`: per-vertex frames by vector diffusion (default), `trivial` or `smoothest`. Its header has the
  measurement: the half-edge frame differs by **79°** between neighbours, against 20-22° for the others.
- `+geom/sphereframe.m:2-40`: **the group gauge in closed form**.
  - The meridian towards +Axis on the registered sphere, pushed forward by 1-ring Jacobians, MaxCondition 50: the same as nxr
    `frames.py` domain "sphere", without a solver. The trivial connection with +1 at both poles *is* the meridian frame (0.36° median).
  - It also gives colatitude and longitude.
  - **Axis=[1 0 0]** gives a second chart whose poles lie on the first chart's equator, for patches near a pole. The two charts differ
    by a known per-vertex rotation.
- `+geom/spherepatches.m`: shared ico-k patches on the registered sphere, with **where the polar gauge breaks**. A patch of half-width ρ
  at colatitude θ sees the frame turn by about 2ρ cot θ relative to transport, so such a patch is excluded or switched to chart 2.
- Ladder: `+geom/surfaceladder.m` (bit-exact to nxr) and `+cores/ladder.m` (two-piece cortex, code 2^ℓ h + t).
  - `+geom/tiles.m`: tile adjacency and centres.
  - `+geom/ladder.m`: the λ range per tile level.
  - `+geom/jointladder.m`: space × time windows.
  - `+ingest/grid.m`: the **time** dyadic grid (level-0 frames of FrameFloor·fs samples, level L = 2^L frames).
- `+operators/connection_laplacian_idt.m`: the same operator as the Studio (r_ji = ψ_ij + π − ψ_ji).
- `+cores/pairgauge.m`: a comparison gauge for vortex pairs (rotate (fa, fb) so fb = 0 at the strongest sensor). It is a
  **different meaning of "gauge"**, internal to pair comparison.

---

## 4. Gaps and conflicts

1. **Five per-vertex frames, three of them arbitrary**:
   - the signpost frame (Studio `tangentBasis` = nxr-neuro `vertexFrames` = nxr `tangent_basis`);
   - gc's `vertexTangentBasis` (nxr-neuro `vertexFramesEmbedded`, used by its VectorHeatSolver and log map);
   - Rheome/Brainstorm (1,1,1)×n (nxr-neuro optical flow, nxr opticalflow);
   - the Studio's x̂-projected `kit.frames()`;
   - the canonical frames (north/west; the only one that means the same across vertices and subjects, and stored per subject).

   Only the canonical frames should ever reach a stored result or a figure. The others are internal bases and must be converted
   before output.
2. **nxr-neuro's log map does not match the Studio's** (the parity rule: a difference is a bug):
   - mesh: input (gc `geometry()`) against intrinsic Delaunay;
   - method: Affine Heat Method by default against heat distance plus vector-heat bearing;
   - θ = 0: gc's reference halfedge against the signpost reference;
   - output: (x, y) against (r, θ).

   nxr-neuro's geodesic distance (`src/geodesic.cpp`) is also on the input mesh. Both go against the maintainers' 2026-11-10 ruling that the
   intrinsic mesh is the default.
3. **Optical flow smoothness**:
   - nxr-neuro, nxr and Rheome use the **gauge-dependent** a'Ka + b'Kb, in a frame that flips across folds;
   - the Studio uses the **gauge-free** connection Laplacian, ruled 2026-10-04 with a sphere-oracle failure of the former.

   These are the same pipeline with different mathematics (a parity breach that predates nxr-neuro). nxr-neuro's own
   `opticalflow_methods.h:15-17` notes that the frame energy "does not charge the frame's rotation at all".
4. **weightedPhaseGradient has no chart and no transport.** It sums ambient 3D P_f, then projects on one plane. That is right only for a
   window much smaller than the fold radius: across a sulcal wall the face vectors point in different ambient directions and cancel.
   The window (which faces) and the axes (e1, e2) are left to the caller, so there is no defined, reproducible chart.
5. **No tiles in nxr-neuro.** There is no `surfaceLadder`, `tileCentres` or roll-up, and no closed-form or Levi-Civita vector
   reductions. Tile summaries of flow cannot be computed in nxr-neuro, and Python still owns them.
6. **No canonical gauge in nxr-neuro.** The trivial connection exists (`computeTrivialConnection`, `trivial`), but there is:
   - no pole placement from the registered sphere;
   - no global-rotation fix (its anchor is vertex 0);
   - no sphere → cortex push-forward;
   - no reader of `<surface>_canonical_frames`.

   nxr's builder still calls the old nxr-compute Node binding.
7. **Anatomical axes are implicit.** On FreeSurfer's sphere, canonical north is the meridian towards +z. If the sphere's axes are
   roughly RAS, north ≈ superior on the sphere (to verify on fsaverage). **"West" (n × north) is not an anatomical axis.** On the
   lateral surface it points anterior on the left hemisphere and posterior on the right (n = −x̂ against +x̂), and it flips again on the
   medial wall. Liu, Wiesman and Baillet (2026) report posterior–anterior / superior–inferior on fsaverage, which is a different and
   hemisphere-consistent frame. Near the poles (precentral, which is active cortex for mu and alpha) the north frame is meaningless
   (`sphereframe.m` warning).
8. **The Levi-Civita and trivial connections disagree by the enclosed curvature.** Closed-form tile sums in canonical frames are exact
   only for the trivial connection. The Levi-Civita resultant needs per-level ρ_ℓ (nxr `levi_civita`). Neither is in nxr-neuro, and the
   choice is not recorded with results.
9. **Tile vector summaries are extrinsic in the Studio and in vortex cores** (ambient means). They are gauge-free, but they cancel
   across folds and are not the same number as a frame-component or transported mean.
10. **Log-map charts are local.** They are valid only to inj(a) ≈ 10 mm on the pial mesh. Any window or chart API must check r < inj(a)
    (or refuse), and windows at larger radii must use the sphere chart.

---

## 5. Proposal for nxr-neuro

One coordinate system, four layers, each named as the Studio, nxr or Rheome names it.

**Layer A: the connection and its frame (`compute/`; mostly exists).**
- Base frame: the signpost frame (`vertexFrames`, = the Studio's `tangentBasis`).
- Connection: the Levi-Civita connection on the operator mesh (`delaunay::connectionLaplacian`).
- Retire `vertexFramesEmbedded` as an output frame (keep it internal to gc calls).

**Layer B: charts (`compute/`, new `nxr/chart.h`, `namespace nxr::chart`).** A port of `surfaceKit.ts` / `nxr.geom.logmap`, all on the
**operator mesh**:
- `heatDistance(m, sources, tCoef=1)`: the Studio's `HeatMethod` (t = tCoef·h̄², h̄ the mean intrinsic edge). It replaces the input-mesh
  gc solver in `geodesic.cpp`, keeping gc's as a named option.
- `transported(m, v, heading)`: (C + M/t) X = M δ e^{iψ}, prefactored, per component, in the signpost frame.
- `logMap(m, v)` → `{r, theta}` exactly as the Studio (θ = arg(∇r / X), 0 along the seed's reference edge), plus `(x, y) = r e^{iθ}`.
  The existing gc `logMap(…, LogMapStrategy)` is kept as `LogMapStrategy::AffineLocal`, a named non-default.
- `gaugeHeading(v, direction)` (nxr `gauge_heading`), so θ is read from canonical north: θ_N = θ − gaugeHeading(v, e1_canonical(v)).
- `injectivityRadius(v)` or a `maxRadius` check (RQ-B's measure), so a chart is refused past the cut locus.
- `disc(v, radius)` (the Studio's `disc`), the window of a chart.

**Layer C: the gauge (`compute/` maths, `pipelines/` I/O), new `nxr/gauge.h`.**
- `sphereFrame(cortex, faces, normals, sphere, axis={0,0,1}, maxCondition=50)`:
  - Rheome's closed form (`+geom/sphereframe.m`) of nxr `hemisphere_frames(domain="sphere")`;
  - meridian on the registered sphere, `jacobians` push-forward;
  - returns e1, e2 = n × e1, singular, colatitude, longitude, north, south.
  - Also `axis={1,0,0}` for the second chart.
  - The trivial-connection solve (`trivial` with ±1 poles plus the log-map rotation fix, nxr `optimal_field`) is kept as the check, and
    as the `domain="cortex"` variant.
- `anatomicalFrame(...)`: e_A and e_S, the sphere's projected anterior (+y) and superior (+z) axes pushed forward the same way, with
  the hemisphere sign made explicit. Pending decision 4.
- `toGauge(z_signpost, frames)` / `fromGauge` (the Studio's `headingOf` / `embedTangent`): convert between the internal signpost basis
  and the canonical frame. Embedded 3V → (c1, c2, cn) is a dot product.
- `pipelines/`:
  - read `<surface>_canonical_frames`, `<surface>_frame_singularities` and `<surface>_sphere` (the Studio adapter's names);
  - when they are absent, compute and store them as Fields and Selections in the surface's folder (schema 50 layout).
  - This removes nxr's dependency on the old nxr-compute Node binding (`frames.py` → nxr-neuro CLI).

**Layer D: dyadic tiles (`compute/` maths, `pipelines/` storage), new `nxr/ladder.h`.**
- `surfaceLadder(positions, faces, maxTiles)`: a bit-exact port of `engine/ladder.ts`, `nxr.geom.ladder.surface_ladder` and
  `+geom/surfaceladder.m`. Graph Dijkstra, stable d_a − d_b sort, `onePiece`, barycentric mass. Golden-tested against nxr's
  `test_atlas_ladder` fixtures.
- `tileCentres(level)`, `labels(level)`, `weylLambda` / `wavelengthOf`, roll-up by shift, and code h·2^D + t (`trees.py`).
- Group tree (fsaverage codes through the registered sphere) after nxr-neuro gains sphere registration (nxr `nxr.register.sphere`).
- **Vector reductions**:
  - `vectorStats` (nxr `vector_stats`: closed-form sums of c1, c2, cn and second moments, in the canonical frame);
  - `leviCivita` / `lcResultant` (nxr `levi_civita`, `lc_resultant`; ρ_ℓ from `transported` at node centres, Ω_ℓ between centres), using
    Layer B's `transported` instead of the Node binding.
- Every stored tile summary records which one it is (`reduction: "frame" | "levi-civita"`) and its frame (`gauge: "canonical frames"`).
- Tiles are **Selections** (the standing rule), never a manifold or folder of their own.

**How the consumers use it.**
- **Optical flow** (`nxr::flow`):
  - add method `connection`: the Studio's `opticalFlow` (vᵀCv on the operator mesh, crossings by `intrinsicGauge`'s
    `intrinsicGradient` / `embedAt`, ported into `compute/`);
  - output: embedded 3V (unchanged), **plus (c1, c2) in the canonical frame** when the subject has one;
  - divergence, curl and speed stay gauge-free scalars, carried to the template as scalars;
  - tile summaries through Layer D;
  - Rheome's a'Ka + b'Kb stays as `method="rheome"` for parity until decision 2 is applied in all three codebases.
- **weightedPhaseGradient**: take an anchor vertex v and a radius R ≤ inj(v):
  - window = `disc(v, R)` faces;
  - each face's P_f is written in its corner vertex's signpost frame, then rotated by the transport angle from v (−arg X);
  - so Σ A_f P_f is a **transported** sum in v's frame, and the direction is reported as a bearing from canonical north at v
    (`gaugeHeading`).
  - The ambient form is kept as `chart="tangent-plane"` for comparison with nxr `phase_gradient_flux`, which must change the same way
    (parity).
- **Vortex-core crossings**: the same Layer D reductions, replacing ambient sums. This applies in all three codebases together.

**Order.**
1. Layer B on the operator mesh (Studio parity tests: r and θ against `surfaceKit.logMap` goldens).
2. Layer C `sphereFrame` plus the reader (parity with `frames.py` / `sphereframe.m` on fsaverage5).
3. Layer D ladder (bit-exact goldens), then the vector reductions. **Done (2026-11-11):** `compute/include/nxr/ladder.h` (ladder, Weyl pairing, subject tree, centres, adjacency; scalar, frame and Levi-Civita summaries via `ChartKit::transported`), golden `ladder_golden.txt` from the Studio's own `ladder.ts` + `surfaceKit.transported` (tile labels identical, areas bit for bit; sums to 1e-15 against nxr's `map_stats` / `vector_stats` / `lc_resultant`, `check_ladder_nxr.py`); `nxr-neuro tiles compute`, and `--tiles` on `opticalflow compute` / `quadrature compute` (`pipelines/include/nxr/pipelines/tiles.h` for the storage).
4. The optical flow `connection` method plus canonical components.
5. The chart-aware `weightedPhaseGradient`.
6. Anatomical frame (after decision 4).
7. Group tree (after sphere registration lands in nxr-neuro).

---

## Open decisions

1. **One output gauge**: every stored or plotted flow direction is in the canonical frames (sphere domain, north/west), with internal
   bases converted. *Recommend: yes.*
2. **Optical-flow smoothness**: make the connection Laplacian (the Studio's, gauge-free) the default in nxr-neuro, nxr and Rheome, and
   keep a'Ka + b'Kb as a named `rheome` method. *Recommend: yes. The current default depends on the (1,1,1)×n frame.*
3. **Log map**: the Studio's construction (intrinsic Delaunay, heat distance plus vector-heat bearing, signpost θ = 0) is nxr-neuro's
   default, and gc's Affine Heat Method is a named option. *Recommend: yes (the parity rule).*
4. **Anatomical axes**: add a stored anterior/superior frame (sphere axes pushed forward, hemisphere sign explicit) beside north/west,
   for reporting as Liu et al. do. *Recommend: yes, as a derived per-vertex rotation of the canonical frame, not a second solve.*
5. **Tile vector summaries**: store both frame-component (closed-form, rolls up) and Levi-Civita (per level) reductions, and stop ambient
   sums in new code. *Recommend: yes, with the reduction named in each result.*
6. **"West" or "east"** for n × north: fix one word across nxr `coords.py`, the spec, Rheome and the Studio. *Recommend: "west" (it is
   correct seen from outside).*

---

## 6. Status: step 2 done (canonical and anatomical frames)

`compute/include/nxr/frames.h` (`canonicalFrames`, a line port of Rheome's `sphereframe.m`, parity ~1e-15 with a MATLAB run; `anatomicalFrame`, `carryAnatomical`, `anatomicalComponents`,
`angleFromAnterior`) and `pipelines/.../frames.h` (`nxr-neuro frames compute`; `opticalflow compute` adds `<stem>_velocity_anterior` / `_velocity_second` when the surface has them).
Words: north / west (nxr `coords.py`'s "east" is the same vector). The anatomical frame is stored per surface as `<surface>_anatomical_{angle,second,handed}` (Fields) and the Selection `<surface>_anatomical_reliable` (angle from north to anterior,
second axis superior | medial with its sign, mask); its record says whether it was carried from a template or taken from the surface's own SCS axes, and from what reference it is measured.
