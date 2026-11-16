# Optical flow on the cortex: the smoothness term, and a bug in Brainstorm's right-hand side

2026-11-11. Findings from comparing nxr-neuro's optical flow with Rheome, Brainstorm (`toolbox/math/bst_opticalflow.m`, Lefèvre 2006–2010) and the code of Liu, Wiesman & Baillet (2026), *Hierarchical Flows of Human Cortical Activity* (bioRxiv; code github.com/Laoma29/Cortical-Flow-Project). Brainstorm's source was read to understand it (the maintainers' decision, 2026-11-11), not to copy it: nxr-neuro's code is written from the maths (GPL).

## The model (all of them)

Horn–Schunck on a surface: per frame, minimise
∫ (∂ₜI + g(V, ∇I))² dA + α ∫ C(V) dA over tangent fields V, with V = Σᵢ (aᵢ e₁ᵢ + bᵢ e₂ᵢ) wᵢ (per-vertex frames e₁, e₂; P1 hat functions wᵢ).
They differ in the smoothness C(V), the data-term discretisation, α, and the input map I.

## The smoothness term C(V)

| Name (nxr-neuro `--regulariser`) | Who | C(V) | Frame-independent | Fold-independent |
|---|---|---|---|---|
| `frame` | Rheome `dynamics.opticalflow_scalar` (and nxr) | cotan Laplacian of the frame COORDINATES: aᵀKa + bᵀKb | **no** | — |
| `ambient` | Lefèvre & Baillet (2008); Brainstorm; the 2026 paper | Σ_{x,y,z} \|∇V_d\|², the Dirichlet energy of V as a 3-vector field (off-diagonal blocks carry the frames' dot products e_k(i)·e_l(j)) | yes | **no** |
| `connection` | the Cortical Flow Studio (`toolbox/flow/opticalFlow.ts`; ruling 2026-10-04 after a sphere test) | \|∇^cov V\|², the connection Laplacian (Levi-Civita, signposts on the intrinsic Delaunay mesh) | yes | yes |
| (Hodge energy) | nxr-neuro `modal` | ‖div V‖² + ‖curl V‖² | yes | yes |

- `frame` ignores how the frames turn between vertices: it depends on the frame choice (Rheome's / Brainstorm's `basis_vertices`, e₁ ∝ (1,1,1)×n) and makes fields constant in those frames free. Rotating the frames' reference axis changes Rheome's velocity by 180 % (measured, PR #30); `ambient` by 8e-16. Kept only for Rheome parity.
- `ambient` = `connection` + \|S V\|² (S the shape operator): a curvature penalty ~κ²\|V\|². On the cortex (radii of curvature 1–2 mm at crowns and fundi) it damps flow at folds and penalises flow ACROSS folds more than along them — a bias from how the sheet is folded in the skull, not from the activity.
- `connection` is intrinsic: unchanged by bending the sheet without stretching. **The default.**
- Bochner: connection = Hodge − ∫K\|V\|² (K Gaussian curvature). The Hodge energy penalises exactly what is measured (div, curl) and leaves shear free.
- α is NOT comparable between regularisers (Rheome's α = 1, the paper's λ = 0.01): choose it per regulariser and report a range.

## The bug in Brainstorm's right-hand side

`bst_opticalflow.m` lines 425–426 (`data_fit_matrix`):

```matlab
B(Faces(:,s)+(k-1)*nVertices) = B(Faces(:,s)+(k-1)*nVertices) + (-1/12 * triangleAreas .* ...);
```

In MATLAB, `B(idx) = B(idx) + x` with a repeated index keeps only the LAST assignment (e.g. `B = zeros(3,1); B([1 1]) = B([1 1]) + [1;2]` gives B(1) = 2, not 3). A vertex is in ~6 triangles, ~2 per corner slot, so B keeps about half of each vertex's contributions — which half depends on the mesh's triangle order. The data and smoothness matrices are built with `sparse` and add up correctly. The correct assembly is `accumarray` (a sum over all triangles).

Checked correct in the same file: the P1 gradients (altitude/altitude²), the triangle areas (true areas), the smoothness and data matrices.

Consequences:
- Brainstorm's velocities are about half the correct ones, by a factor that varies from vertex to vertex (valence, face order). nxr-neuro's `ambient` (correct sum) gives speeds ~2.2× Brainstorm's on the same mesh and maps, direction cosine 0.97 (PR #30's golden).
- **The 2026 paper's code has the same lines** (`data_fit_matrix.m`, lines 20–21): its flow magnitudes are about halved and its kinetic energy |V|² about quartered, with a mesh-dependent spatial pattern; directions are slightly perturbed. Its direction-based results (posterior→anterior, alignment with the hierarchy) are probably robust, not exact.
- To reproduce Brainstorm's or the paper's numbers, the last-write behaviour must be imitated (not implemented as of 2026-11-11). Reporting it upstream is the maintainers' call.

## Other differences from the paper

| | Liu, Wiesman & Baillet 2026 | nxr-neuro (2026-11-11) |
|---|---|---|
| Source model | LCMV beamformer, depth-corrected, overlapping spheres; one value per source | MNE free orientation (any stored kernel) |
| Input map | the band-filtered SIGNED source signal (1–13, 13–30 Hz, broadband; zero-phase FIR), scaled by its max \|·\| | `--input amplitude` (the wavelet member's \|K c\|, default) or `--input signed` (Re K c, fixed kernel) |
| Smoothness / α | ambient (with the RHS bug), λ = 0.01, not varied | connection (default), α to be chosen and varied |
| Data term | per triangle (FEM) | `--datafit triangle` (default) or `vertex` (Rheome's) |
| Outputs | kinetic energy, direction in an anatomical frame on fsaverage, alignment with the Margulies gradient, kinetic states | divergence, curl, speed, the Hodge–Dirac split |

Tracking the signed signal follows the moving oscillation (mostly phase); tracking the envelope follows amplitude, whose in-place growth and decay brightness constancy turns into divergence (the 99.7 % exact fraction on the first real run).

## Removed

- The `frame` regulariser and the `vertex` data term (Rheome parity) are gone from the code, the CLI, the tests and `opticalflow-compare.sh` (the `rheome` row). `connection` (default) and `ambient` remain.
- `modal --modal-energy frame` is gone. `--modal-energy connection` (default) charges the connection-Laplacian energy of `direct` / `pcg` on the modes' flows (full K x K, each mode's embedded flow read in the connection's gauge by `ConnectionFlow::gaugeOf`); `diagonal` / `full` (Hodge energy) stay. With every mode on a sphere, modal + connection differs from direct + connection by 7.3e-2 (relative, max over frames) in the unit test: the modes' span (the glued potentials miss two of the 2V dimensions) and the gauge reading of an embedded vector.
- `quadrature::weightedPhaseGradient` takes a `ChartKit`, an anchor and a log-map disc radius; each face's P is carried to the anchor by the vector heat transport before summing.
- Modal + connection energy now also uses the connection flow's own gauge data term (gauge-averaged gradient, vertex-lumped, the flow's mass); with every mode on a sphere it differs from direct + connection by 4.4e-2 (was 7.3e-2), the modes' span and the gauge reading. The modes' small system gets a 1e-9 ridge (a combination of modes with no gauge image is otherwise left to rounding). `diagonal` / `full` keep the ambient vertex-averaged data term.

- **Update: the `modal` method is removed** (on the real cortex it reproduced neither the connection flow's speed nor its maps even at 1,000 modes, and was slower than `direct`). The lines above about `modal` are kept as the record. The Hodge-Dirac reading of the full velocity gives the modal decomposition.
