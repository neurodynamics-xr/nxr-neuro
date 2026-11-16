# Operators and eigenmodes across nxr, Rheome, the Studio and nxr-compute: differences and naming (2026-11-10)

The rule says nxr (Python), Rheome (MATLAB), the Cortical Flow Studio (TypeScript) and nxr-compute (C++, now `nxr-neuro/compute`) implement **the same pipelines and the same mathematics**.

Method: one read-only inventory per codebase, each citing file and line, cross-compared here. Paths are abbreviated as follows.
- nxr = `nxr/src/nxr/`
- Rheome = `rheome/+rheome/`
- Studio = `nxr-cortical-flow/studio/src/`, with `D/` = `engine/toolbox/differential/`
- compute = `nxr-neuro/compute/`, with `H` = `include/nxr/compute.h` and `S/` = `src/`

Legend: ✅ the same in all four · ⚠️ a real difference in the numbers · 🏷️ a naming difference only · 🐛 a bug.

---

## 1. Which operator exists where

| Operator | nxr | Rheome | Studio | nxr-compute |
|---|---|---|---|---|
| **Laplace–Beltrami (cotan)** | `cotan_laplacian` on the intrinsic mesh. Also `meshops.laplacian` (GᵀAG) on the embedded mesh. | `operators.laplace_beltrami` on the embedded mesh. Also `intrinsic_delaunay` `.L`. | `cotanLaplacian` on the intrinsic mesh | `LaplacianCotan`. Intrinsic only if `Manifold(intrinsicDelaunay=true)`, and never mollified. |
| **DEC d0, d1, ⋆0, ⋆1, ⋆2** | inside the Hodge–Dirac solve, on the intrinsic mesh | inside `operators.hodge_dirac`, on the intrinsic mesh. Rheome's registry says "there is no d0/d1 here" 🐛. | `dec()`, intrinsic. No production caller. | `Dec`, **embedded always**, ⋆0 = A/3. Plus a separate intrinsic set inside `hodgedirac::assemble`. |
| **Hodge–Dirac d+δ (pencil K, M)** | `pencil` (tests only); the production solve uses the two ladders | `operators.hodge_dirac` | `hodgeDirac()`. No production caller; offline scripts only. | `HodgeDirac` (`S/hodge_dirac.cpp`) |
| **Connection Laplacian** | `connection_laplacian`: intrinsic, signposts, **complex** n×n | two versions. `connection_laplacian`: embedded, **complex**, Galerkin mass. `connection_laplacian_idt`: intrinsic, **real 2n, interleaved 2×2**. | `connectionLaplacian`: intrinsic, **real 2n, interleaved 2×2** | `LaplacianConnection`: complex, or `Real2N` as **block [re; im]**, plus ε = 1e-8·I. Also face, Crouzeix–Raviart and trivial-gauge versions. |
| **Relative Dirac L(τ)** (Liu–Jacobson–Crane) | — (none) | `operators.dirac_frame` / `eigen.dirac_frame` (the Brainstorm `bst_dirac` port) | `relativeDirac(cotan, E, τ, s)` | `dirac(τ)` / `diracFamily_`; `OperatorId::Dirac` = **the E block only** |
| **First-order extrinsic Dirac D** (Gauss map, 4F×4V) | — | — (Rheome's `dirac_extrinsic` is **E = DᵀWD**, not D) 🏷️⚠️ | `diracD` (inside `relativeDirac.ts`) | `DiracD` = registry `extrinsicDirac` |
| **First-order intrinsic Dirac D_int** (edge vectors) | — | `dirac_intrinsic` (+ `dirac_intrinsic_sq`) | — (read from compute) | `DiracIntrinsicD` = `intrinsicDirac` |
| **Face Diracs D̃, D̃_int, Ẽ, L̃(τ)** | — | — | — | yes |
| **Spin Diracs** (face spinors; extrinsic, Hoffmann–Ye, Ye) | — | `operators.spin_dirac`, `eigen.spin_dirac` | `spinDirac.ts`. No production caller. | `ops::spin`, `eigenSpin` |
| **Covariant / Weitzenböck / connection gradient** | — | — | — | yes (compute only) |
| **Face gradient, div, curl** | `FaceOperators` (float32 positions) **and** `face_gradient`/`differential.*` (float64, the Rheome port). A duplicate copy also sits in `inverse/fused.py`. | `operators.face_gradient`, `differential.*`, `weak_differential` | `assembleFaceGradient` (float32) | `facegrad::gradient`/`laplacian` (Green–Gauss on faces, a different operator) |
| **Heat method, vector heat, signed heat** | `geom/heat`, `logmap`, `signedHeatDistance`; **and** a separate Rheome-port `heatDistance` with its own Delaunay | `heatDistance` with its own Delaunay | `HeatMethod`, `#vectorHeat`, `signedHeat` | geometry-central wrappers (embedded) |

---

## 2. Mesh: intrinsic Delaunay or embedded

| Operator | nxr | Rheome | Studio | nxr-compute |
|---|---|---|---|---|
| Hodge–Dirac (all forms) | intrinsic, mollified ✅ | intrinsic, mollified ✅ | intrinsic, mollified ✅ | its own intrinsic, mollified ✅ |
| Laplace–Beltrami | intrinsic, mollified | **embedded**, no flips ⚠️ | intrinsic, mollified | intrinsic **only with the flag, unmollified** ⚠️ |
| Connection Laplacian | intrinsic + signposts | **embedded** (the complex one) / intrinsic (`_idt`) ⚠️ | intrinsic + signposts | intrinsic with the flag (no mollify); trivial-gauge **embedded** |
| Relative Dirac, its Δ part | — | **embedded** | intrinsic, mollified | intrinsic with the flag, else embedded |
| Relative Dirac, its E part | — | embedded | embedded, unmollified | embedded |
| DEC (stand-alone) | n/a | n/a | n/a | **embedded always**, even with the flag, so `d0ᵀ⋆1d0 ≠ cotanL` there 🐛 |
| Face gradient / div / curl | embedded ✅ | embedded ✅ | embedded ✅ | embedded ✅ |

**The mollify rule** is the same everywhere: δ = 1e-6·mean edge length, and a uniform ε is added to every edge if it is positive (Sharp & Crane 2020). There are exceptions:
- compute's `Manifold(intrinsicDelaunay)` doesn't mollify at all;
- the `heatDistance` copies in nxr and Rheome use different flip tolerances (−1e-12 vs −1e-6), a LIFO order, and a barycentric mass *after* the flips.

**The number of Delaunay implementations:**
- nxr: 3 (`intrinsic_delaunay`, `heatDistance`, `flipOut`);
- Rheome: 2;
- Studio: 1;
- compute: 2 (geometry-central's signpost flips, used both for the flag and inside Hodge–Dirac).

---

## 3. Mass matrices

| Mass | nxr | Rheome | Studio | nxr-compute |
|---|---|---|---|---|
| **Mixed Voronoi (Meyer)** of the mollified *input* mesh, recorded before the flips | `lumped_mass`: the Hodge–Dirac ⋆0, heat, `hat_operators` | `intrinsic_delaunay` `S.mass`: the Hodge–Dirac ⋆0, `connection_laplacian_idt` | `lumpedMass`: **default everywhere** in the kit, the Hodge–Dirac ⋆0, connection (×2), relative Dirac (×4) | the Hodge–Dirac ⋆0 only |
| **Galerkin** (consistent) | — none | **default**: LB, connection, `dirac_frame` (⊗I₄), `lb_connectome`, Poisson, every import | — none | **default** for the LB and relative-Dirac eigenproblems |
| **Barycentric A/3** | `meshops`, `ladder`, `vortex.Cortex.mass`, mean curvature, register/tangent | `mass(…,'lumped')` (opt-in), mean curvature | `ladder.ts`, `curlCoreTiles`, mean curvature | geometry-central's `lumped` **and** DEC ⋆0 |

**Name clash** 🏷️: "**lumped mass**" means *mixed Voronoi* in nxr and the Studio, and *barycentric A/3* in nxr-compute and geometry-central. Rheome's `hodge_dirac` header says "star0 = lumped mass" but uses the mixed Voronoi 🐛 (doc). compute's docs once called A/3 "Voronoi".

### The consequence: "Laplace–Beltrami modes" are three different discretisations ⚠️

| | nxr / Studio | Rheome `eigen.modes` (`import/bases`) | nxr-compute `eigenOperator(LaplacianCotan)` |
|---|---|---|---|
| mesh | intrinsic, mollified | embedded | intrinsic only with the flag, unmollified |
| mass | mixed Voronoi | Galerkin | Galerkin (default) |
| surface | the glued whole cortex | per hemisphere | as given |

### Relative Dirac: three versions ⚠️

| | Rheome `dirac_frame` | Studio `relativeDirac` | compute `dirac(τ)` |
|---|---|---|---|
| Δ part | embedded cotan | intrinsic, mollified | the flag's cotan (OPG) |
| E part | embedded, **area-weighted** normals | embedded, **angle-weighted** normals | embedded, the Gauss map (vertex normals) |
| scale | divides each part by its **largest generalised eigenvalue** (sL, sE; Brainstorm), so the spectrum is in [0, 1]; `normalize=false` gives physical units | s = `diracScale` = 2A/∫\|dN\|² | s = 1 |
| mass | Galerkin ⊗ I₄ | mixed Voronoi × 4 | Galerkin ⊗ I₄ (default) |
| τ | 0.5 (hard-coded in callers) | `DIRAC_TAU` = 0.5 | argument |
| domain | **per hemisphere** | per component | as given |

The Studio's parity test with compute holds only with s = 1 and compute's embedded cotan, not with what the Studio actually uses 🐛 (claimed parity).

---

## 4. Hodge–Dirac: the one place they mostly agree, and where they don't

**The same in all four ✅:**
- pencil K = [[0, ⋆1d0ᵀ…]] with M = diag(⋆0, ⋆1, ⋆2), on the glued closed surface, intrinsic and mollified;
- ⋆0 = mixed Voronoi of the input mesh;
- ⋆2 = 1/area;
- solved as two ladders: exact Δ0 = d0ᵀ⋆1d0 against ⋆0, and coexact Δ2 = ⋆2d1⋆1⁻¹d1ᵀ⋆2 against ⋆2;
- **one λ cut** at the smaller top of the two ladders;
- μ = √λ;
- kinds 0 (exact) and 2 (coexact);
- the stored package P = ψ/√2 (exact) and Wχ/√2 (coexact, face→vertex averaged);
- **no eigenvector sign convention anywhere.**

**Differences:**

| Item | nxr | Rheome | Studio (offline scripts) | nxr-compute |
|---|---|---|---|---|
| ⋆1 floor (cocircular edge → M ≤ 0) | **none** 🐛 (`1/s1` unguarded) | floor 1e-6 (Rheome PR #94) | **none** 🐛 (M singular if ⋆1 = 0) | ✅ none by default; optional `star1Floor` |
| Shift | σ = −1e-3 absolute | σ = −1e-3 absolute | σ = −1e-3 (ladders); the full pencil uses **σ = 1.0** while the docstring says "nearest 0" 🐛 | ✅ **since 2026-11-10 (PR feat/hodge-reference-method): σ = −1e-3 absolute, one solve per ladder, as nxr/Rheome**; the old relative ladder (σ = −1e-7·λmax, −1e-4·λmax) and slicing are named options (`shift`, `slice`) |
| Constants dropped | λ > 1e-6·max (relative) | the same | Δ2: relative; LB: **λ > 1e-6 absolute** ⚠️ | ✅ λ > 1e-6·max per ladder, as nxr (was over both) |
| Re-orthonormalisation | none (ARPACK's) | none (eigs's) | none | ✅ none by default; `normalize` (Cholesky whitening) is an option |
| Output | the vertex package | the vertex package | the vertex package | full form vectors (ψ, d0ψ/μ, 0)/√2; no vertex package |
| Harmonic kind 1 | — | — | — | defined (`kind` 0/1/2), but the split never sets it |

**Two problems that affect us directly:**
1. 🐛 **nxr-compute's sliced solve is wrong.** `test_hodge_dirac` "sliced (30/window) == single window" fails: max Δλ 14.2, and two returned vectors nearly parallel (M-orthonormality 0.95). Likely cause: the first window sets U exactly at a converged eigenvalue, without the 0.98 margin later windows use, so the next window shifts onto it and a multiplet member is counted twice while another is lost.

   `hodgeDiracModes` slices whenever the count is over 400. That means **any real-cortex solve** through nxr-compute (e.g. 3000 modes) is affected, and so would be nxr-neuro's `eigen solve`. It has to be fixed first.
2. ⚠️ **Coexact normalisation.** The coexact vertex potentials aren't orthonormal under the vertex mass (Studio measurement: squared norms 0.44–0.71). nxr (`fold`, `fused_weights`) and the Studio's `hodgeProjection.ts` correctly use the Gram matrix. The Studio's `hodge.ts` brush (`a = 2ge…`, `weightsFromCoefficients`) assumes ⟨P, P⟩ = ½ for every mode.

---

## 5. Other eigen solves

| | nxr | Rheome | Studio | nxr-compute |
|---|---|---|---|---|
| Generic solver | `eigsh` shift-invert (Hodge only) | `eigen.smallest` (the Brainstorm `bst_eigs_smallest` port): σ = −1e-7·λmax, then −1e-4·λmax, then `smallestabs` | no runtime eigensolver (modes-free Chebyshev/LDLT); offline SciPy | `eigenSmallest` = the same Brainstorm ladder ✅ with Rheome |
| Degeneracy fix | none | `eigen.modes`: Cholesky Rayleigh–Ritz | n/a | `normalize` whitening; Dirac quartets rebuilt (`reconstructMultiplets`) |
| Known-bad shift still in use | — | `gauge` "smoothest" and `geom.tree` call `smallestabs` directly, the pattern `modes.m` documents as wrong (~40 % residual) 🐛 | — | the JS and MATLAB wrappers still default to **σ = −1e-8**, the documented-bad fixed shift 🐛 |
| `removeDC` | per ladder | keeps the DC mode | — | removes exactly **one** mode (wrong for two hemispheres or a quaternion 4-fold zero) 🐛 |

---

## 6. Naming: one concept, many names

| Concept | Names in use |
|---|---|
| **d + δ basis** | Hodge–Dirac · Hodge-Dirac · `hodgeDirac` (stored names, functions) · `hodge-dirac` (operator kinds: `"hodge-dirac weights"`, JSON `operator`) · `ops::hodgedirac` · coordinate-system key **`eigenvalue:hodge-dirac`** (nxr `io/layout.py`) vs the written **`eigenvalue:hodgeDirac`** 🏷️ · brush `operator: 'hodge'` |
| **Relative Dirac** | relative Dirac · `dirac_frame` (Rheome, both packages) · bare **"Dirac"/"dirac"** (Rheome `forward.dirac`, `inverse.dirac`, `source.dirac`; the Studio's kernel domain `'Dirac'`; compute `OperatorId::Dirac`, which is **only E**) · L(τ) · `relativeDirac` |
| **First-order extrinsic Dirac** | `DiracD` / `extrinsicDirac` / `diracD` / "Crane Dirac" · Rheome's **`dirac_extrinsic` is E, not D** · the Studio adapter's `'dirac-extrinsic'` (γ) is **built from compute's `diracIntrinsicD`** 🏷️⚠️ |
| **Laplace–Beltrami** | LBO · LB · `laplace_beltrami` · `laplaceBeltrami` · `LaplacianCotan` · `cotanLaplacian` · cotan · L · K · stiffness · Δ₀ · L0 · `'lb'` · `lb3000` |
| **Connection Laplacian** | connection Laplacian · vector Laplacian · Levi-Civita · CL · C · `'cl'` · Bochner · `leviCivitaConnectionLaplacian` |
| **Mass** | lumped (two meanings, §3) · Galerkin · Voronoi (old misnomer) · ⋆0 · `hodge0` · `S.mass` · `star0` |
| **Eigenvalue** | λ (second order, 1/m²) · μ (first order, 1/m) · γ (the Studio's extrinsic Dirac) · ν (1/(λ−σ) or 1/μ) · Rheome `dirac_frame` returns **`Lambda`** for a second-order operator · cortical-flow `diracD_mu` is an eigenvalue of **E** (second order) 🏷️ · units 1/m vs 1/mm (`vortex.py`, `envelope.py` positions in mm) |
| **Mode kind** | 0 exact / 2 coexact everywhere; compute also defines **1 harmonic** |
| **"ladder"** | three meanings: the exact/coexact eigen ladders; the dyadic tile/scale ladder (`geom.ladder`, `ladder.ts`); compute's **shift ladder** (`kShiftLadder`) 🏷️ |
| **"normalize"** | three meanings in compute: M-orthonormalise eigenvectors; "normalized mesh" = intrinsic Delaunay; `DelaunayNormalization` = extrinsic `fixDelaunay` 🏷️ |
| **K̃** | two operators in compute: `faceLaplacianGreenGauss` and `faceLaplacian2Form` 🏷️ |
| **Layouts** | complex as complex n×n (nxr, Rheome `connection_laplacian`) · real 2n **interleaved 2×2** (Studio, Rheome `_idt`) · real 2n **block [re; im]** (compute `Real2N`) · quaternions interleaved 4v+c (all) · compute's covariant / Weitzenböck **component-major** c·N+v ⚠️ |

---

## 7. Other bugs and stale claims found

- 🐛 **nxr:**
  - face-operator defaults disagree: `divergence(tangential=False)` vs `curl(tangent=True)` vs `div_curl("tan")`;
  - stale docstring `register/modes.py:17` → `hodge_dirac.solve_input` (gone);
  - `geom/heat.py` docstring says t = 2·mean², but the code uses 1·mean².
- 🐛 **Rheome:**
  - `+operators/Contents.m` omits about 10 operators;
  - the registry omits Hodge–Dirac, `spin_dirac`, the `_idt` connection Laplacian, `lb_connectome` and `connectome_laplacian`;
  - the CHANGELOG and the `intrinsic_delaunay` header say mollification and signposts are "not ported", but both are in the code;
  - `spin_dirac.m` cites the non-existent `spin_vertex_to_face`.
- 🐛 **Studio:**
  - `export-subject-primary.mts` calls `glueMesh(P, F, 0.003)`, an old signature;
  - the `ModesFreeFrame` area for CL and Dirac is 2A and 4A, so its level j is a different λ band from the scalar kernels'.
- 🐛 **nxr-compute:**
  - `-Wswitch` in `facets.cpp`: `HodgeDirac` and `SpinDirac` have no cache case;
  - `EigenResult` says "sorted ascending", but `bothSigns` returns positives then negatives;
  - the registry says Hodge–Dirac is `Domain::vertex` / `scalar`;
  - CONTRIBUTING.md marks Weitzenböck "planned" though it's built;
  - stale names `solveEigenmodes`, `assembleMeshOperators`, `ComputeContext`;
  - the drift-guard `assert` is compiled out in release builds.

---

## 8. Decisions

**Approved:** items 1–3 below. Order of work: **fix the bugs first, then consolidate the names.** Item 3's three choices are still open; item 4 isn't decided yet.

### Eigenvalue convention: λ and μ (approved)

An eigenvalue is named by the **order** of its operator:

| Symbol | Operators | Equation | Values | Units |
|---|---|---|---|---|
| **λ** | second order, Laplace-type: Laplace–Beltrami, connection Laplacian, Hodge Laplacians Δ₀ Δ₁ Δ₂, relative Dirac L(τ), and E = DᵀWD | L φ = λ M φ | λ ≥ 0 | 1/m² |
| **μ** | first order, Dirac-type: Hodge–Dirac d+δ, first-order extrinsic and intrinsic Diracs D and D_int, spin Diracs | D ψ = μ M ψ | signed, in ±μ pairs | 1/m (a wavenumber) |

- **The link:** a Dirac operator squares to a Laplacian, D² = Δ, so **μ² = λ**. For the Hodge–Dirac this is exact: an exact mode with Laplace–Beltrami eigenvalue λ sits at μ = √λ, and a coexact mode at μ = √λ of the 2-form Laplacian.
- **Wavelength:** always **2π/√λ = 2π/|μ|**.
- **Usual convention:** λ is the usual Laplace–Beltrami eigenvalue axis (Weyl λ_k ≈ 4πk/A, the heat kernel e^{−tλ}, Meyer frame edges 4π2ʲ/A). μ for first-order operators is our convention: the literature often writes λ there too.
- **Fixes this rules out:**
  - Rheome's `dirac_frame` `Lambda` is correctly λ, but only with `normalize=false`. With Brainstorm's default scaling, its spectrum is a unitless value in [0, 1] and must be named as normalised, not λ.
  - cortical-flow's `diracD_mu` is an eigenvalue of E (second order), so it is a λ.

## 9. Proposed single conventions (the maintainers approved 1–3; Rheome code ported into nxr-neuro follows them)

1. **Names (one per concept, used in all four):**
   - `hodgeDirac` (d+δ);
   - `relativeDirac(τ)`;
   - `diracExtrinsic` (first-order D, Gauss map);
   - `diracIntrinsic` (first-order D_int);
   - `spinDirac{Extrinsic, IntrinsicHY, IntrinsicYe}`;
   - `laplaceBeltrami`;
   - `connectionLaplacian`;
   - bare "Dirac" retired;
   - λ for second-order, μ for first-order eigenvalues;
   - kinds 0/1/2;
   - "ladder" kept for the exact/coexact ladders only.
2. **Mesh and mass (a scientific choice):** the convention all four already share for Hodge–Dirac, the **intrinsic mollified Delaunay mesh with the mixed-Voronoi mass**, becomes the default for every operator: LB, connection, and the relative Dirac's Δ. Galerkin and barycentric stay available as named options, never as silent defaults. "Lumped" is renamed to what it is (`massMixedVoronoi` vs `massBarycentric`).
3. **⋆1 floor (1e-6) everywhere** (nxr and the Studio lack it).
4. **Relative Dirac:** one definition. Pick the scale (Brainstorm's sL/sE, the Studio's 2A/∫|dN|², or 1), the normals (area- or angle-weighted) and the domain (per hemisphere or glued).
5. **Complex layout:** one rule for real-2n storage (interleaved 2×2, as the Studio and Rheome) at every binding edge.
6. **Fix before anything else:** compute's sliced solve; the bad σ = −1e-8 defaults; `removeDC`; the ⋆1 floors.
7. **Coexact normalisation:** the Gram matrix everywhere (fix the Studio's `hodge.ts` brush).

---

## 10. Status in nxr-neuro (branch `feat/one-delaunay`, 2026-11-10)

Steps 1, 2 and 5 of the order of work, for nxr-neuro:
- **One Delaunay construction** (`compute/include/nxr/delaunay.h`, `src/delaunay.cpp`), a line port of the Studio's `intrinsicDelaunay.ts`, replaces geometry-central's `SignpostIntrinsicTriangulation` and `mollifyIntrinsic` everywhere in `compute/`.
- **Default for every operator**: the intrinsic mollified Delaunay mesh with the mixed-Voronoi mass of the mollified input (recorded before the first flip): cotan Laplacian, the three masses, the stand-alone DEC (`d0ᵀ⋆1d0 == cotanL`, `hodge0 == mass`), the vertex connection Laplacian (the signposts' gauge) and the Hodge–Dirac pencil. `mesh = intrinsic|embedded` (`intrinsicDelaunay`) and `mass = mixedVoronoi|galerkin|barycentric` are named options; `Lumped` is renamed `Barycentric` (no alias).
- **Named, recorded constants** `mollify` (1e-6) and `flipEps` (1e-6), in the library, the bindings, `nxr-neuro eigen solve --mollify --flip-eps`, the provenance and the cache key. The pipeline now mollifies exactly as nxr does.
- **Parity**: `compute/test/parity/test_delaunay_parity.py` (Studio `tsx` and nxr `prepare`).
- Still on the input mesh (geometry-central), by design of this step: the Hodge decomposition of a 1-form and the trivial connection (`Manifold::decEmbedded()`), the face Diracs, the extrinsic Dirac E, the Face/Edge connection domains. The relative Dirac's Δ part now follows the mesh option (intrinsic by default); its mass default is the mixed Voronoi mass. Open (later steps): the ⋆1 floor everywhere (item 3), the relative Dirac's single definition (item 4), the Rheome ports into nxr-neuro.
