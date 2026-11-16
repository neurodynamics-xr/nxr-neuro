# The structural connectome on the cortex (`compute connectome`)

Status: **spec, for the maintainers' review.** No code yet. Asked for by the maintainers (2026-11-16): the structural connectome is a native
computation of nxr-neuro whenever a subject has fibres, or can borrow a template's; nothing calls MATLAB or Brainstorm, and there are no wrappers.

**Reference:** the maintainers' `tess_connectome` (MATLAB), the reference. The default of every option reproduces it; every departure is a named
option recorded in the provenance and the cache key (the method rule, `CONTRIBUTING.md`). Its helpers come from Brainstorm (GPLv3): none is ported;
each is replaced by an existing nxr-neuro function or written from the mathematics (section 4).

## 1. Why

Activity on the cortex can move in two ways: along the sheet, from one patch to its neighbours (the paths the mesh and its operators describe), and
across it, through the white-matter fibres that join distant regions. A travelling, rotating or standing pattern measured on the sheet alone cannot
say which of the two carries it. The structural connectome is the second set of paths, written on the same cortex and the same vertices as every
other measurement: W(u, v) says how many fibres join the place u to the place v.

With it, the measurements of spatiotemporal patterns can be put against anatomy:
- whether a pattern's path, or the coupling between two regions, follows the fibres between them or only the sheet;
- the connectome's own spatial modes (graph harmonics of W) as a basis beside the mesh's eigenmodes;
- the coupling of homologous regions of the two hemispheres, which are joined by fibres and not by the sheet;
- the same comparisons at any declared scale: a vertex, a tile of the dyadic ladder, a region of a parcellation.

What it is: streamline counts from tractography, placed on the cortex by the fibres' ends. A subject without its own tractography borrows a
template's, carried onto its cortex through the registration sphere: then W is the template's anatomy on the subject's cortex, not the subject's
own, and the provenance says so (`registration`).

## 2. Inputs

All inputs are the caller's, named explicitly. There is no default for which data is used.

| input | what | where it comes from (io, by name) |
|---|---|---|
| **the surface** | the cortex W is written on: positions [V × 3] and faces [F × 3] | a surface (manifold) of the subject, by name |
| **the fibres** | a fibre bundle: nF polylines of nP points, SCS metres | the subject's own bundle (`curve/<fibers>`, written by `convert` from a Brainstorm fibres file), or a bundle of the template subject (e.g. the default subject) |
| **the template surface** (template fibres only) | the template's cortex on which the template fibres' ends are placed | a surface of the subject that holds the fibres, by name |
| **the registration spheres** (template fibres only) | `<surface>_sphere` of both surfaces (FreeSurfer's `sphere.reg`), one vertex per cortex vertex, the pieces (left, right) as contiguous runs | stored with each surface, as `frames compute` and the group tree read them |
| **the parcellation** (`--resolution roi` only) | a partition: one region code per vertex, the region names from its dictionary | a stored atlas Selection of the surface, by name |

**Type checks before anything runs** (preparation; each refusal names its reason): the fibres and both surfaces are in the same coordinate system
(SCS, metres); `nP ≥ 2`; each sphere has as many vertices as its surface and the same pieces, in the same order (left, right), as the other sphere;
the parcellation is a partition of THIS surface; the fibres belong to the subject (own) or to the subject's template (`template_id`); fibres of
another subject that is not the template are refused.

**Own or template fibres.** The reference looks for the subject's own fibres and falls back on the default anatomy's. nxr-neuro does not choose:
`--from` names the bundle. A bundle of the template subject REQUIRES `--template-surface`; a bundle of the subject itself refuses it. When the
template surface IS the target surface (computing on the template itself), no registration is made, as in the reference.

**Options** (names as `tess_connectome` names them; the options struct uses snake_case, the command line kebab-case, the provenance both):

| option | reference name | default (the reference's) | meaning |
|---|---|---|---|
| `resolution` | `Resolution` | `vertex` | `vertex`: a vertex × vertex connectome · `roi`: region × region over a parcellation |
| `atlas` | `Atlas` | the reference's default is `Desikan-Killiany`; here REQUIRED with `roi` (it names data: open question 3) | the parcellation of `roi` |
| `smooth_hops` | `SmoothHops` | `3` | the hops p of the mesh smoothing (`vertex`); 0 = none |
| `edge_thr` | `EdgeThr` | `1e-4` | edges below `edge_thr · max(W)` are removed (`vertex`); 0 = none |
| `registration` | (none: new) | `shepard` (the reference's) | how template ends reach the subject: `shepard` (section 3.4) · `nearest` (nxr's group-tree correspondence, a named alternative) |

## 3. The graph

Every node is a pure function of matrices in `compute/` (matrices in, matrices out; callable with arrays made anywhere; no file, no subject, no
knowledge of where the fibres came from). Preparation reads the inputs and type-checks them; the graph runs once (there is no recording, so no
chunk loop); readback hands the outputs to io.

    fibres [nF·nP × 3] ──fibreEnds──▶ end points [2nF × 3]
        (template fibres)  ──nearestVertex(template surface)──▶ template ends [2nF] ──carryEnds(match)──▶ ends [2nF]
        (own fibres)       ──nearestVertex(surface)──────────────────────────────────────────────────▶ ends [2nF]
            match = strongestDestination(shepardWeights(subject sphere, template sphere, pieces, K = 8))     (registration = shepard)
                  | sphereNearest(template sphere, subject sphere, pieces)                                     (registration = nearest)

    vertex:  ends ──endConnectome──▶ A [V × V] (self pairs dropped) ──sparseCongruence(Sᵖ)──▶ Sᵖ A Sᵖᵀ ──symmetrize, offDiagonal──▶ W₀
             W₀ ──measure max──▶ m ;  W₀, edge_thr·m ──thresholdBelow──▶ W ──componentsOf──▶ labels ──measure count──▶ sizes ──largest──▶ keep
             where S = smoothingOperator(faces) = D⁻¹(Adj + I)
    roi:     end points (own) | positions of ends (template) ──nearestCentroid(regionCentroids(surface, partition))──▶ region per end [2nF]
             ──regionPairs──▶ (a, b) per fibre ──measure streamlines over the region × region partition──▶ C [R × R]

### 3.1 The ends (`fibreEnds`, `nearestVertex`)

- `fibreEnds(points, nP)`: the first and the last point of each fibre, [2nF × 3] (rows f and nF + f, the importer's order). Pure indexing.
- `nearestVertex(positions, query)`: for each query point the vertex at the smallest Euclidean distance; the lowest index on a tie. The reference
  is MATLAB's `dsearchn`. Exists as `nxr::crossings::nearestVertex` (brute force); moved to a general home (`compute/include/nxr/nearest.h`) and
  given a k-d tree (Bentley 1975) for 10⁵ fibres on 10⁴–10⁵ vertices; same answers, tested against the brute force.
- The ends are stored as the Studio's bundle ends: int32 [2nF], fibre f's ends at 2f and 2f + 1, `-1` for an end on no vertex (none here: every
  end has a nearest vertex; `-1` appears only if a later option gates ends by distance).

### 3.2 The vertex connectome before smoothing (`endConnectome`)

A(u, v) = the fibres with one end at u and the other at v, symmetric:

    A = E₁ᵀ E₂ + E₂ᵀ E₁,   E₁, E₂ [nF × V] the one-hot rows of each fibre's first and second end.

The reference then drops the diagonal (a fibre whose two ends fall on one vertex adds 2 at (u, u), removed). This is the field the importer already
stores (`<fibers>_connectome`) minus its diagonal; it is also the Studio's vertex connectome `A` (`connectome.ts`), whose `tileConnectome`
counts a fibre inside one node ONCE on the diagonal. Returned as an Eigen sparse matrix (duplicates summed, as MATLAB's `sparse`). The sum over
fibres is the definition of the field (the sparse product above, general algebra), not a reduction over time or cortical space; see open question 6.

### 3.3 The mesh smoothing (`smoothingOperator`, `sparseCongruence`)

Raw ends leave most of the cortex without an edge: tractography stops at the grey/white boundary, so the ends cluster on gyral crowns and miss the
sulcal banks. The reference bridges this with a diffusion on the mesh:

- `smoothingOperator(faces, V)`: Adj the binary vertex adjacency of the mesh's edges (both directions, a vertex's own row excluded), and
  S = D⁻¹ (Adj + I), D = diag(degree + 1): a row-stochastic one-hop average of a vertex with its neighbours. Written from the definition (the
  random-walk normalisation of a graph); the edges come from the faces as `ladder::tileAdjacency` reads them.
- Sᵖ for p = `smooth_hops` by p sparse products (p = 0: the identity).
- `sparseCongruence(P, A)` = P A Pᵀ, P and A sparse: a new general-algebra function beside `nxr::algebra::congruence` (which is dense, on a pattern).
  This is the Studio's `frameConnectome` C_ψ = Ψᵀ A Ψ with Ψ = (Sᵖ)ᵀ: an end at vertex a is spread over the vertices i with weight Sᵖ(i, a).
- then W = (W + Wᵀ)/2 and the diagonal removed (`offDiagonal`), as the reference.

Units: streamlines, spread over the mesh (fractional counts). The total is not conserved: the columns of a row-stochastic Sᵖ do not sum to 1 on an
irregular mesh. This is the reference's definition, kept.

**Memory (stated in the options struct).** Sᵖ holds ~|p-ring| non-zeros a row (about 37 at p = 3 on a regular mesh); W holds up to nF·|p-ring|²
non-zeros before the threshold (10⁵ fibres: ~10⁸, ~1.6 GB as float64 CSC with int64 indices). The congruence is formed column block by column
block and thresholded before the next block is formed, so the peak is the kept W plus one block; the threshold needs max(W) first, so the first
pass measures the max per block and the second forms and thresholds (two passes, the same result as the reference's one-shot product).

### 3.4 Template fibres onto the subject (`shepardWeights`, `strongestDestination`, `carryEnds`)

The reference places each template end on the template surface, then sends it to the subject vertex that the registration associates most strongly
with that template vertex:

1. `nearestVertex(template positions, end points)` → a template vertex per end.
2. `shepardWeights(dest sphere, src sphere, pieces, K)` → a sparse interpolation [V_subject × V_template], piece by piece (left to left, right to
   right): for each subject vertex, its K = 8 nearest template vertices on the registered sphere (unit-normalised; the weights are scale-free), with
   squared chord distances d₁ ≤ … ≤ d_K and the modified Shepard weights of Franke & Nielson (1980) on them:
   w_k = ((d_K − d_k) / (d_K d_k))², k = 1 … K − 1, normalised to sum 1 (the K-th neighbour sets the radius and takes no weight; a zero distance is
   replaced by the smallest positive double; a vertex whose K − 1 weights are all 0 gives its nearest neighbour weight 1). Written from the paper's
   formula, with these choices stated; never from Brainstorm's code.
3. `strongestDestination(Wmat)` → for each template vertex, the subject vertex with the largest weight in its column (the lowest index on a tie).
   A template vertex that no subject vertex counts among its neighbours has an empty column: the reference then sends it to subject vertex 0 (the
   first index MATLAB's `max` returns). The default reproduces this; open question 1.
4. `carryEnds(templateEnds, match)` → the subject vertex of each end. The reference then re-snaps the end to the nearest subject vertex of that
   vertex's position, which returns the same vertex; the native chain uses the index directly.

`registration = nearest` (named alternative): the template vertex's sphere point → the nearest subject sphere vertex, piece by piece: the
correspondence of the group tree (`nxr::ladder::groupTreeFor`, nxr's `Template.tree_for`) and of nxr's template tractography (`pipelines/atlas/hcp.py`).
It has no empty columns.

### 3.5 Threshold and the largest component (`thresholdBelow`, `componentsOf`, `largest`)

- `m = max(W₀)` over all entries: a measurement (`max`) of the field over the one-member partition, read back before the threshold (the
  reductions rule). If W₀ has no entry, m = 0 and nothing is removed (the reference: `W < 0` is never true).
- `thresholdBelow(W₀, t)`, t = `edge_thr` · m: entries below t set to 0 and dropped from the pattern. Pointwise.
- `componentsOf(W)`: the connected components of the graph whose edges are W's non-zeros, over all V vertices (an isolated vertex is its own
  component). This is `nxr::regions::detectRegions` with one value per vertex (1), the non-zeros as pairs and enter = exit = 1: components numbered
  by their lowest vertex (MATLAB's `conncomp` numbers them in the same order).
- sizes: the vertex count of each component, a `count` measurement over that labelling.
- `keep`: the vertices of the largest component; on a tie the lowest-numbered (the reference's `max` over `tabulate`). W itself is NOT restricted
  to `keep` (the reference returns the whole W and the indices beside it).

### 3.6 The region connectome (`regionCentroids`, `nearestCentroid`, `regionPairs`)

The reference's `roi` path does not use the vertex connectome:
- `regionCentroids(positions, codes, R)`: the unweighted mean position of each region's vertices (the reference averages vertex positions; not
  `ladder::tileCentres`, which is area-weighted and returns a vertex). A region with no vertex is refused.
- `nearestCentroid(centroids, end points)`: each end goes to the region whose CENTROID is nearest (Euclidean, the lowest region on a tie), not to the
  region that holds its nearest vertex. Own fibres use their raw end points; template fibres use the subject positions of their carried ends.
- `regionPairs`: each fibre's two regions; fibres with both ends in one region are left out (the reference: `asg(:,1) ~= asg(:,2)`).
- C(a, b) = the fibres between regions a and b, symmetric (C + Cᵀ), zero diagonal: the `streamlines` measurement over the region × region
  partition, the rows the importer writes for Brainstorm's own region connectomes (`code`, `within_code`).
- Named alternative `assign = vertex` (open question 4): an end takes the region of its nearest vertex, which is the Studio's `tileConnectome`
  P ᵀ A P and works for any partition, the dyadic tile ladder included (and then rolls up the ladder by `rollUp`).

### Checked against the Studio's chain

The Studio's `connectome.ts` has the same pieces, and they are ported with its names: the bundle's `ends` (int32, two a curve, −1 off the
surface), the vertex connectome `A` as a pass over the curves, `tileConnectome` (Pᵀ A P), `rollUp` (a coarser ladder level as a sum of
2ˢ × 2ˢ blocks) and `frameConnectome` (Ψᵀ A Ψ). The smoothing is a `frameConnectome` with sparse atoms; the threshold and the components are new
nodes the Studio does not have. No stage needs the Studio to change.

## 4. The Brainstorm helpers, and what replaces each

`tess_connectome` calls these Brainstorm functions (and MATLAB built-ins). None is ported.

| called by the reference | what it does there | native replacement | status |
|---|---|---|---|
| `in_tess_bst` | loads the surface (vertices, faces, atlases) | io: the surface read by name (`read_surface`), the atlas Selection by name | existing |
| `bst_get('SurfaceFile' / 'Subject', 0)` | finds the subject, its fibres, the default anatomy | io: the subject's stored nodes and its `template_id`; the bundle named by `--from` | existing |
| `file_fullpath`, `load(...,'Points')` | reads the fibres file | io: the curve manifold `curve/<fibers>` and its `points` (written by `convert` from the Brainstorm file, natively) | existing |
| `tess_interp_tess2tess` | the subject × template interpolation on the registered spheres, per hemisphere | `shepardWeights` (new, from Franke & Nielson 1980) over the stored spheres `<surface>_sphere`, the pieces from `nxr::ladder::pieceRuns` | new |
| ↳ `bst_shepards`, `bst_nearest` (inside it) | modified Shepard weights on the K nearest | `shepardWeights` with a k-nearest query (new: the k-d tree of 3.1) | new |
| `fibers_helper('AssignToScouts')` | each end → the nearest scout centroid | `regionCentroids` + `nearestCentroid` (new; the nearest query of 3.1) | new |
| `dsearchn` (MATLAB) | nearest vertex | `nearestVertex` (existing `nxr::crossings::nearestVertex`, moved and given a k-d tree) | existing, extended |
| `sparse`, `accumarray`, `spdiags`, `mpower` (MATLAB) | the sparse algebra | Eigen sparse (triplets sum duplicates) in `endConnectome`, `smoothingOperator`, `sparseCongruence` (new, general algebra) | new |
| `graph`, `conncomp`, `tabulate` (MATLAB) | components and the largest | `nxr::regions::detectRegions` (existing) + a `count` measurement + `largest` | existing |

## 5. What is stored

Names and kinds are those the importer writes for the same thing, so a reader tells a native connectome from an imported one only by its producer.

**`resolution = vertex`:**
- a Field of kind `connectome`, SPARSE (CSC: `indptr`, `indices`, `data` float32 as the importer's, or float64: open question 5) on the product
  manifold surface × surface, at `source-source/<name>`; unit `streamlines`; the description says smoothed over `smooth_hops` hops and
  thresholded at `edge_thr` of the max.
- `<name>_component`: a Selection (indices) of the largest component's vertices (`keep`), as a reliability mask is stored, its `params_json`
  naming the connectome field.
- `<name>` defaults to `<fibers>_connectome_s<p>` (own) or `<fibers>_on_<surface>_connectome_s<p>` (template), so the importer's unsmoothed
  `<fibers>_connectome` is never overwritten; `--name` sets it.

**`resolution = roi`:** `streamlines` measurement rows on the atlas Selection (`code` a, `within_code` b, value the count; both orders, as the
importer writes Brainstorm's region connectomes), OF a byte-less Field `<name>` of kind `connectome` that carries the provenance (the precedent of the
`tile map` fields: rows of a computed measure point to a Field). The region labels are the atlas dictionary's.

**Provenance** (`producer_json`, and every option in the cache key): `method: "tess_connectome"`, `reference: "tess_connectome (MATLAB)"`,
`resolution`, `atlas`, `smooth_hops`, `edge_thr`, `registration` (`none` | `shepard` | `nearest`), `fibers` (the bundle's subject, name and id),
`n_fibers`, `surface` and `template_surface` (ids), `n_edges` (upper triangle), `n_keep`, and, under the reference's own names, `Resolution`,
`Atlas`, `SmoothHops`, `EdgeThr`. A rerun with the same inputs and options is a cache hit; `--replace` recomputes. No instrument warnings are
written.

The schema `io/schema/nxr.connectome.schema.json` (a connectome under `manifold/` with `keep` as an array) has no writer; open question 2.

## 6. The command line

Verb first, as the command line is moving to (`docs/specs/2026-11-16-distribution-and-interoperability.md`, section 7):

    nxr-neuro compute connectome <surface> --from <fibers> [--template-surface <surface>]
                                 [--resolution vertex|roi] [--atlas <atlas>] [--smooth-hops 3] [--edge-thr 1e-4]
                                 [--registration shepard|nearest] [--name NAME] [--replace]
                                 --dataset D/N --subject S        (until the context of section 7, point 2, exists)

- `<surface>` and `--from` are REQUIRED: which cortex and which fibres are the caller's. `--from <fibers>` names a bundle of the subject;
  `--from <template>:<fibers>` a bundle of its template subject, and then `--template-surface` is required.
- `--atlas` is required with `--resolution roi` and refused with `vertex`; `--smooth-hops` and `--edge-thr` are refused with `roi`.
- Until the verb-first grammar lands, the same verb is also `connectome compute` (today's noun-first form); one parser, one dispatch.
- Output: JSON lines (progress: `ends`, `registration`, `connectome`, `components`; then `"stage": "result"` with the names written, `n_edges`,
  `n_keep`, `cache`).

Files: `compute/include/nxr/connectome.h` (+ `nearest.h`, `shepard.h`, the sparse congruence in `congruence.h`),
`pipelines/include/nxr/pipelines/connectome.h` (`ConnectomeOptions`, `connectome_compute(ds, sub, opt, progress)`), `cli/connectome_command.cpp`,
and the method's section in `docs/reference.md`.

## 7. Tests

**Unit tests, one a node** (`compute/test/test_connectome.cpp`, synthetic arrays only):
- `fibreEnds`: the first and last rows, order f then nF + f.
- `nearestVertex` (k-d tree): equal to the brute force on random clouds; the lowest index on exact ties.
- `endConnectome`: symmetric; Σ A = 2 × (fibres with distinct end vertices); a self-pair leaves no entry; two fibres on one pair give 2.
- `smoothingOperator`: rows sum to 1; p = 0 gives the identity; on a regular lattice Sᵖ's support is the p-ring.
- `sparseCongruence`: equals the dense P A Pᵀ on small random sparse matrices; symmetric for symmetric A.
- `thresholdBelow`, `offDiagonal`: by hand on a 4 × 4.
- `componentsOf` + `largest`: two blocks of different sizes; a tie goes to the lowest; isolated vertices are components.
- `shepardWeights`: rows sum to 1; a destination on a source point gives weight 1 to it; the K-th neighbour has weight 0; invariant to the sphere's
  radius; pieces never mix (a left vertex never weights a right one).
- `strongestDestination`: ties to the lowest; an empty column gives 0 (the reference's behaviour).
- `regionCentroids`, `nearestCentroid`, `regionPairs`: by hand; same-region fibres left out.

**The golden against the reference** (`compute/test/golden/connectome_golden.txt`, generator `gen_connectome_golden.m` beside it). The generator
is run ONCE by a developer in MATLAB with the maintainers' Brainstorm (the one holding `tess_connectome`), never at run time and never by a test.
It builds a throwaway Brainstorm protocol with SYNTHETIC data only, calls `tess_connectome`, and writes its inputs and outputs as text:
- **the cortex:** two icospheres (642 vertices each, radius 80 mm, centres 100 mm apart) as left and right pieces, their `Reg.Sphere` the unit
  icosphere positions, a synthetic atlas of 8 regions a piece (by longitude), so the hemispheres and regions are exact.
- **own fibres:** 400 fibres of 20 points between random pairs of points placed 1–3 mm off the surface (straight polylines with a random bend),
  some within one hemisphere, some across, a few with both ends at one vertex and a few duplicated; points drawn so that no end is within 10⁻⁶ m of
  a tie between two vertices or two centroids.
- **template fibres:** a template subject (the default anatomy of the protocol) with its own two-icosphere cortex at another resolution
  (2562 vertices a piece) and its sphere randomly rotated by a few degrees per piece, so that `tess_interp_tess2tess` does real work, and with a
  denser template than subject so that some template vertices have empty columns.
- **cases:** own · vertex at `SmoothHops` 0, 1, 3 and `EdgeThr` 0 and 1e-4; own · roi; template · vertex (default options); template · roi;
  template fibres on the template surface itself (no registration).
- **written:** the meshes, spheres, atlas codes, fibre points, each case's options, W as sorted triplets (i < j, value to 17 significant digits),
  `keep`, C and its labels, and, as intermediates the generator reads directly from the Brainstorm functions, `tess_interp_tess2tess`'s matrix as
  triplets and the template-vertex → subject-vertex map.
- `test_connectome_golden`: ends, assignments, `keep`, C and the registration map identical; W's pattern identical and its values to 1e-12
  relative (the sum order of the sparse products differs from MATLAB's).

**Pipeline test** (`pipelines/test/test_connectome_pipeline.cpp`, registered in `pipelines/CMakeLists.txt`): a synthetic dataset with a subject, its
template, the surfaces, spheres, an atlas and fibres written through io. Own and template fibres, both resolutions; the stored field equals the
compute chain's W; with `--smooth-hops 0 --edge-thr 0`, the stored W equals the importer's `<fibers>_connectome` minus its diagonal; a rerun is a
cache hit and a changed option a miss; refusals (no `--from`; template fibres without `--template-surface`; a sphere of the wrong size or pieces;
a missing atlas; `--atlas` with `vertex`; a bundle of an unrelated subject; mismatched coordinate systems).

Tests run through `tools/tests.py run connectome`, scoped to these.

## 8. Open questions for the maintainers

1. **Empty columns of the registration.** With `registration = shepard`, a template vertex that no subject vertex counts among its 8 nearest
   neighbours sends its fibre ends to subject vertex 0 (the reference's `max` over an all-zero column). This happens whenever the template is denser
   than the subject. Reproduce it by default (the method rule), and add a named option (`unmatched = nearest`: the nearest subject vertex on the
   sphere), or correct the reference itself and regenerate the golden?
2. **One storage for connectomes.** The importer writes a sparse `connectome` Field on the product manifold; the schema
   `nxr.connectome.schema.json` (W under `manifold/`, `keep` as an array) has no writer. This spec follows the importer and stores `keep` as a
   Selection. Retire the schema, or write it as well?
3. **`--atlas` without a default.** The reference defaults to Desikan-Killiany. Here the atlas is data, so it is required. Keep it required, or
   accept the reference's name as the default when the surface holds an atlas of that name?
4. **Region assignment.** The reference assigns an end to the nearest region CENTROID, which can place an end in a region that does not contain its
   nearest vertex (long or curved regions). Add `assign = vertex` (the region of the nearest vertex: the Studio's `tileConnectome`, and the dyadic
   tile ladder with `rollUp`) as a named option now, or later?
5. **Precision of the stored W.** float32 as the importer's counts (exact for integer counts) or float64 (the smoothed weights are fractional)?
6. **The sum over fibres.** W = E₁ᵀE₂ + E₂ᵀE₁ sums over the fibre set. This spec treats it as the definition of the field (sparse algebra, as the
   importer and the Studio do), not as a reduction over time or cortical space. Agreed, or should the fibre set be a partition and the count a
   `streamlines` measurement over it?
7. **Template fibres in the default subject.** nxr-neuro's default subject does not yet carry a template bundle (`docs/reference.md`, "Not done
   yet": HCP-1065). Is importing one (nxr's `fibre/hcp1065` and its ends) part of this work, and onto which template surface: the template's pial,
   or a white surface where tractography ends (nxr places the template ends on a white surface)?
8. **Distance gate.** The reference assigns every end to a vertex however far it is; nxr's template import drops ends farther than 5 mm from the
   cortex. Add a named option (`gate`, metres; default none, the reference's)?
