# The magnetic Laplacian: the phase summary over time, trials and subjects (spec, 2026-11-12; APPROVED by the maintainers the same day)

**Rulings on approval (2026-11-12):**
- The magnetic Laplacian and its eigenmodes come first; storage, the steps and the time route below wait (the compute functions come first, PR #92).
- The edge weights are not the Laplacian's concern:
  `magneticLaplacian` takes the complex edge weights W; how W is made from data (cross-power, coherency, an imaginary part) is decided later, outside it.
  Decisions 2 and 3 below are therefore set aside.

## Why

The phase of an oscillation goes round the circle at its own frequency, so instantaneous phase maps cannot be averaged: over time every vertex's phase
smears to nothing. What stays stable is the **lag between neighbours**: the pairwise product z_u z̄_v ignores the pattern's common rotation and can be
averaged over time, trials or events. The averaged lags then need not fit together (noise, mixed patterns, rotation), so one map has to be **solved** from
them. Writing each vertex as a complex number x_u and asking x_u ≈ e^{iθ_uv} x_v on every edge gives, once the |x_u| = 1 constraint is relaxed, the
quadratic form of the **magnetic Laplacian**; its lowest eigenvector is the best-fitting map (Singer's angular synchronisation). The pointwise phase
gradient (`phase_current`, `phaseGradient`) stays the tool for a single instant; this is the tool for a summary.

What the eigenpairs give:
- **the lowest mode's phase:** a latency map (time = φ / ω): sources earliest, sinks latest; its gradient gives direction and speed;
- **its magnitude:** where that consensus holds (small where the lags are incoherent or contradict each other): a confidence map;
- **its zeros:** points where the magnitude vanishes and the phase winds a full turn: vortex cores;
- **λ₁, the frustration:** the misfit no single map can remove; near 0, one travelling wave explains the lags; larger, rotation or competing waves.
  One number per subject, band (and window);
- **higher modes:** further phase-consistent patterns, orthogonal to the first: a basis for complex cortical data that follows the measured flow.

**Scope:** the method knows only its input, the averaged edge lags. Nothing in it knows the instrument, the kernel or where the phase
came from; the choice of edge weight is an OPTION a pipeline sets ("we are building a toolbox").

## The maths

On the mesh's edge set E (each undirected edge once), with S the averaged edge cross-products (S_uv = ⟨z_u z̄_v⟩, Hermitian: S_vu = conj S_uv) and
the vertices' own power P_u = S_uu:

    W_uv = w_uv e^{iθ_uv}            the edge's complex weight (the edge option below)
    D_uu = Σ_v |W_uv|                its degree
    L = D − W                        the magnetic Laplacian: Hermitian, positive semi-definite; its quadratic form Σ_E |W_uv| |x_u − e^{iθ_uv} x_v|²
    normalised: L_n = D^{-1/2} L D^{-1/2}   (Singer's form: the lowest eigenvector of L_n is the synchronisation estimate)

**The edge option** (`edge`), from S, every one a named option (the definition sets it):
- `cross_power`: W_uv = S_uv;
- `coherency`: W_uv = S_uv / √(P_u P_v) (|W| ≤ 1);
- `imaginary_coherency`: **to be defined with the maintainers (decision 2):** the imaginary part of coherency is a real, signed number, so it cannot by itself
  be a U(1) weight. Candidates: (a) θ_uv = arg(γ_uv) with weight w_uv = |Im γ_uv|; (b) θ_uv = arcsin(Im γ_uv) (the lag whose sine is the imaginary
  coherency), weight 1 or |γ_uv|.

**The other options:** `normalise` (`none` · `symmetric`, Singer's) and `modes` (how many eigenpairs, k).

## The graph (the paramount architecture)

**The spectral route first** (first). No recording is read; the inputs are stored nodes.

| part | what | where |
|---|---|---|
| loading | the band's stored sensor cross-spectrum C (an operator of kind `cross-spectral density`, `csd compute`), a FIXED-orientation kernel K (one row a source), the kernel's target surface | io: the finder |
| preparation | type checks: C complex M × M on the kernel's channels; K scalar (free-orientation kernels refused: decision deferred by the maintainers, below); the surface's edges | pipelines: the plan |
| graph | `edge_cross_spectrum`: S_uv = K_u C K_vᴴ on the edges and the diagonal (**`congruence(A, M, pattern)`**, the general A M Aᴴ on a sparsity pattern, compute) → `magnetic_laplacian` (W from S by the edge option; L) → `hermitian_eigs` (the lowest k eigenpairs) → `winding` (the winding number of a complex vertex field round each face) | compute |
| readback | the modes, the eigenvalues, the power P, the winding per face | io: stored as below |

**The time route next:** the recording through the chunk service; `band_filter`/`analytic` → `apply_kernel` (fixed) → `edge_product` (z_u z̄_v per
edge per sample: POINTWISE on the edge pattern) → `measure` (the mean over a time-tile partition: ARRAY-BACKED ROWS on the tile Selection, the maintainers'
answer, 2026-11-12, as Rheome's atlas measurement) → per window, the same `magnetic_laplacian` → `hermitian_eigs`; λ₁ per window a series of rows on the
same partition. It needs the runner's measurement outputs (rows instead of Fields), built with it.

## The compute functions (pure: matrices in, matrices out; each with a unit test and a reference)

- `congruence(A, M, pattern)`: (A M Aᴴ) on the pairs in `pattern` only, A [n × m] real, M [m × m] complex Hermitian: O(|pattern| m²), never the n × n
  product. General algebra, its own function (the architecture policy).
- `magneticLaplacian(edges, S, P, options)`: the Hermitian sparse L (or L_n) from the edge values and the diagonal, by the edge option.
- `hermitianEigs(L, k, sigma)`: the lowest k eigenpairs of a sparse Hermitian matrix: Spectra 1.2.0's `HermEigsSolver` with shift-invert through
  Eigen's complex `SimplicialLDLT` (the scratch check, PR #90: a 20,000-vertex ring with flux, within 7.5e-9 of the exact spectrum, natively and in
  WebAssembly). Degenerate clusters closed under the solver's known miscount as the Dirac solve does; a dense oracle (Eigen's
  `SelfAdjointEigenSolver<MatrixXcd>`) for small meshes, for tests only.
- `winding(x, faces)`: the winding number of arg x round each face (the sum of the wrapped phase differences along its edges / 2π): −1, 0, +1 for a
  vortex core's face.

## What is stored

- **the eigen axis:** a points manifold (coordinate system `eigenvalue:<stem>`, positions the eigenvalues λ₁ … λ_k), as the Hodge–Dirac modes are;
- **the eigenvectors:** an operator of kind `eigenvectors` from the axis to the surface, [V × k × 2] (re, im: the cross-spectrum's complex layout);
- **the vertex power** P (a scalar Field) and **the winding** of the first mode (a Field on the faces, cell `2`);
- **the provenance:** the cross-spectrum and kernel ids, the band, every option, the solver's tolerance and convergence, the multiplets; the cache key
  the same.

## Tests (synthetic only; scoped)

- `congruence` against the dense A M Aᴴ on random small matrices;
- `hermitianEigs`: the ring with flux α (exact λ_k = 2 − 2cos(2π(k + α)/n)); random sparse Hermitian matrices against the dense oracle; parity with
  scipy's `eigsh` (ARPACK, complex Hermitian) through nxr's environment (`*/test/parity/`);
- the method: a plane wave on a sphere (its exact lags on the edges): the lowest mode's phase is the wave's phase up to a constant, λ₁ ≈ 0, the
  magnitude uniform; a rotating wave (a lag that winds round the sphere): λ₁ > 0 and the winding finds the cores; two waves mixed: λ₁ grows with the
  mix;
- the pipeline: the shipped definition on the synthetic subject, end to end, against the same computation done directly.

## What it cannot resolve (into `producer_json.warnings` and `docs/reference.md`)

- The phase map is defined **up to a global rotation**: only phase differences are meaningful (latency relative to the earliest point).
- Lags of more than half a cycle between neighbours are aliased (the mesh must sample the wave: edge length ≪ wavelength).
- Where the magnitude is small, the phase is not determined: read it only with the magnitude.
- λ₁ compares across subjects only on the same mesh and options (it scales with the edge weights and the number of edges unless normalised).

## Open decisions

1. **Approve the spec** (the graph, the functions, the storage, the order: spectral route, then the time route).
2. **`imaginary_coherency`:** which construction (a) or (b) above, or another, is "the imaginary part" you intend to use.
3. **Defaults:** none for `edge` (a definition must name it: it chooses which data the method reads), and the method's reference for the rest:
   `normalise: symmetric` (Singer), `modes: 10`. Agreed?
4. **Free-orientation kernels are refused** until the free-orientation question is taken up again.

## Order of PRs

1. `congruence` and `hermitianEigs` (compute, with their tests and the scipy parity).
2. `magneticLaplacian` and `winding` (compute), with the plane-wave and rotating-wave tests.
3. The steps (`edge_cross_spectrum`, `magnetic_laplacian`, `hermitian_eigs`, `winding`), the storage of complex eigenmodes, the shipped definition
   (`magnetic_phase_map`), `docs/reference.md` section.
4. The time route: `edge_product`, the runner's measurement outputs (array-backed rows on the tile partition), λ₁ per window.

## Notes (2026-11-12, after #92): the magnetic Laplacian as a connection Laplacian; the lag field as a 1-form; free orientation

**The same operator as the connection Laplacian, with a measured connection.** Both are L = D − W, W_uv = w_uv U_uv, U_uv a unit complex number that
transports a value from v to u; both have the quadratic form Σ w_uv |x_u − U_uv x_v|² and a first-order root d x = x_u − U_uv x_v with L = dᴴ diag(w) d
(compute's `connectionGradient` d^∇ is the geometric one).

| | connection Laplacian (Levi-Civita / trivial) | magnetic Laplacian |
|---|---|---|
| x | a tangent vector a vertex, a complex number in its frame | a complex number a vertex (phase, amplitude) |
| U_uv | e^{i n ρ_uv}: the frames' rotation (geometry) | e^{iθ_uv}: the measured lag (data) |
| w_uv | cotan | the caller's |
| holonomy round a loop | the enclosed Gaussian curvature | the sum of the lags: the wave's circulation |
| λ₁ = 0 when | the connection is flat (never for Levi-Civita on a sphere) | the lags are consistent: one map explains them |
| point singularities | the trivial connection's (index sum χ) | vortex cores (`winding`) |

So the magnetic Laplacian is a connection Laplacian on the complex line bundle whose U(1) gauge field is measured; λ₁ measures how far that connection
is from flat (the frustration). Consequences to take up when the operator is wired in:
- **the operator registry:** the magnetic Laplacian listed with the connection Laplacians (bundle: complex line; holonomy: measured; its root d^θ);
- **base weights:** `graph` or `cotan` (a geometry-aware fit), as the connection Laplacian has them; the transport stays the data's.

**The lag field is a 1-form on the edges** (θ_uv, antisymmetric), so it has a Hodge decomposition (compute's `hodgeDecompose`):
- its **exact** part is the gradient of a latency potential: the travelling wave the lowest eigenvector recovers;
- its **coexact** part is rotation: the vortices, and the source of the frustration;
- its **harmonic** part is the circulation the surface's topology forces.
For small lags the eigenproblem and the exact part (a Poisson solve) agree; the eigenproblem stays right when the lags wrap round the circle. A
decomposition of the measured lags is a readout of its own: how much of the propagation is travelling and how much rotational.

**Free-orientation currents (deferred by the maintainers; the route this suggests):** a 3-vector a vertex needs, to be moved to its neighbour, both the geometry's
transport (the frames' rotation and the normal's tilt: compute's flat full-frame transport `Fⱼᵀ Fᵢ`) and the data's lag (a phase). Their product is a
transport in U(1) × SO(3), which a QUATERNIONIC connection carries at once: the vector-valued magnetic Laplacian is a connection Laplacian on the
quaternionic bundle, where the Hodge–Dirac and relative-Dirac operators already live.

## Direction needs a global sign (2026-11-12)

The magnetic Laplacian reads DIRECTION (a phase map, a latency), so its input must carry a sign that is consistent across the cortex. Findings:
- **Region collapses do not give one.** MNE's `pca_flip` / `mean_flip` (`region::collapse`, `pca_flip`) extract a region's signal without cancellation,
  each region signed by its own majority orientation: arbitrary against its neighbours. On small tiles adjacent across folds, a large share of the lags between
  neighbours can come out near π (a bimodal distribution), and the imaginary coherency, which ignores 0 and π, is then at the noise level.
- **The standard ROI measures avoid the question** by being sign-invariant (|coherency|, |Im γ|, PLI, wPLI: how strongly, not which way). Direction
  (signed lag, phase, latency, dPLI) is sign-sensitive.
- **A plain average over a region** cancels across folds and smooths phase towards zero; it is not offered (`region::collapse` has no `mean`).
- **Therefore, before any directional map:** a global sign, solved jointly (Z₂ synchronisation, the real twin of this operator; it also erases genuine
  anti-phase) or removed (free orientation: the quaternionic route). And every lag readout reports its distribution (the share near π), never a median
  alone, with the agreement of vertex-level and region-level maps.

## The transported magnetic Laplacian (2026-11-12, built: PR #98)

The free-orientation route to a global sign: the current a 3-vector, compared after carrying it along the surface (`transport::leviCivita` or `frames`,
`transport::compare`), and the operator L = D − W with W_uv = w e^{iθ} R_uv (`magnetic::transportLaplacian`, 3n x 3n). Under Levi-Civita transport the
NORMAL line is carried exactly onto itself on any surface, so a normal current with consistent lags is a null vector everywhere (no π at folds), while
tangential fields feel the curvature's holonomy: on a curved surface the normal current is the unique lowest mode (tested on a sphere). On a flat piece
of surface the parallel tangential fields are null vectors too: the phase map is determined, the vector's direction among them is not, without the data's
own local direction (an option for later).

## Decision: direction analysis makes no orientation assumption

The currents are what the inverse says they are: the aim is to analyse and describe the lags. The normal orientation of cortical currents is a prior,
not a fact; the method does not assume it.

**The route:** direction analysis should make no orientation assumption. Keep the full free estimate, three components per source, and let the data say
how the currents at two places relate. That's the joint data-block approach: each edge carries the pair's full 3 × 3 cross-spectrum block, and the
magnetic Laplacian fits one complex 3-vector per vertex consistently over the whole graph. Orientation and lag come out together, with no prior on
either, and only one global phase is left.

- The geometry is used for what it is sure about: distances along the surface, scales, tiles, neighbourhoods; not orientation.
- The geometry transport (`transport::leviCivita`) encodes the normal-current prior; it stays a named option, not the default and not this route.
- A fixed (normal-constrained) kernel imposes the same prior on the estimate itself; the direction route uses the free estimate.

## References

- A. Singer, *Angular synchronization by eigenvectors and semidefinite programming*, Appl. Comput. Harmon. Anal. 30 (2011) 20–36.
- M. Fanuel, C. M. Alaíz, J. A. K. Suykens, *Magnetic eigenmaps for community detection in directed networks*, Phys. Rev. E 95 (2017) 022302.
- M. Cucuringu, *Sync-Rank: robust ranking, constrained ranking and rank aggregation via eigenvector and SDP synchronization*, IEEE Trans. Netw.
  Sci. Eng. 3 (2016) 58–79.
- Spectra 1.1.0's `HermEigsSolver` (Y. Qiu), and Eigen's complex `SimplicialLDLT`.
