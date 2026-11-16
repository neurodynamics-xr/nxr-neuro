# The structural connectome on the cortex (`compute connectome`)

Status: **spec, for the maintainers' review.** No code yet. Revised 2026-11-16 to follow the maintainers' decisions of that day; each is marked
"decided 2026-11-16".

**Reference:** the maintainers' `tess_connectome` (MATLAB), the reference. The default of every option reproduces it, except where a decision below
says otherwise and the spec names the difference (section 8). Every departure is a named option recorded in the provenance and the cache key (the
method rule, `CONTRIBUTING.md`).

## 1. Why

Activity on the cortex can move in two ways: along the sheet, from one patch to its neighbours (the paths the mesh and its operators describe), and
across it, through the white-matter fibers that join distant regions. A travelling, rotating or standing pattern measured on the sheet alone cannot
say which of the two carries it. The structural connectome is the second set of paths, written on the same cortex as every other measurement.

With it, the measurements of spatiotemporal patterns can be put against anatomy:
- whether a pattern's path, or the coupling between two regions, follows the fibers between them or only the sheet;
- the connectome's own spatial modes (graph harmonics of the connection weights) as a basis beside the mesh's eigenmodes;
- the coupling of homologous regions of the two hemispheres, which are joined by fibers and not by the sheet;
- the same comparisons at any declared scale: a vertex, a tile of the dyadic ladder, a region of a partition.

What it is: fiber counts from tractography, placed on the cortex by the fibers' ends. A subject without its own tractography uses the template's,
carried onto its cortex through the registration: the result is then the template's anatomy on the subject's cortex, not the subject's own, and the
provenance says so.

## 2. What was decided (2026-11-16)

1. **Native compute** (decided 2026-11-16), for subjects whose packages hold fibers. Nothing calls MATLAB or Brainstorm at run time. The helpers the
   reference takes from Brainstorm (GPLv3) are replaced natively, written from the mathematics and never ported: the Shepard weights from Franke &
   Nielson (1980), the nearest-vertex search, connected components, Eigen sparse algebra.
2. **The pointwise object is each fiber's pair of end vertices** on a chosen surface (decided 2026-11-16). Counting fibers per pair of places is a
   **measurement over a partition**, reduced at readback without materialising the vertex matrix (section 4). The vertex connectome is the finest
   partition, materialised only on request. Tiles are stored at the finest level and rolled up at read, as the measurement pyramid does.
3. **Smoothing over neighbouring vertices is linear, so it is pushed onto the partition's memberships** (decided 2026-11-16): smooth the memberships,
   then count. The smoothed vertex matrix is never formed for a partition result. Pruning weak connections and keeping the largest component apply to a
   materialised connectome only.
4. **Partitions, not "atlas" or "roi"** (decided 2026-11-16): a fiber end belongs to the region of its vertex. The reference's nearest-centroid
   assignment is dropped; region results therefore differ from the reference's wherever an end lies nearer another region's centre than its own
   region's (section 8).
5. **The reference's quirk is fixed** (decided 2026-11-16): a template vertex with no subject neighbour sends its ends to subject vertex 0. Here such
   an end goes to its nearest subject vertex on the registration sphere (section 5.1).
6. **One storage form; no older connectome schema** (decided 2026-11-16, section 6).
7. **Precision follows the input** (decided 2026-11-16): no silent change of precision anywhere in the chain (section 5.4).
8. **Descriptive names** (decided 2026-11-16): "fiber counts" for the raw counts, "connection weights" for the smoothed and pruned result. Single
   letters are never names; a symbol appears only inside a formula whose terms are defined in words.
9. **Template fibers** (decided 2026-11-16): the template package carries HCP-1065 (section 7).
10. **The distance gate** (decided 2026-11-16): `--max-end-distance`, default 5 mm (section 5.2).
11. **Golden tests** (decided 2026-11-16): section 9.

## 3. Inputs

All inputs are the caller's, named explicitly. There is no default for which data is used.

| input | what | where it comes from (io, by name) |
|---|---|---|
| **the fibers** (`--fibers`, required) | a fiber bundle: nF polylines of nP points, SCS metres | the subject's own bundle (`curve/<fibers>`), or a bundle of its template (`<template>:<fibers>`, section 7) |
| **the end surface** (`--end-surface`, required) | the surface on which each fiber's ends are placed: a location on the cortex, not a claim about the depth at which a fiber ends. The pial surface is the surface source maps are written on; others are allowed | a surface by name; for template fibers, the same name on the template and on the subject |
| **the partition** (`--over`, required) | what the counts are reduced over: `vertices`, a tile level of the dyadic ladder, or a stored partition Selection of the end surface (a parcellation) | by name |
| **the registration** (template fibers only) | the subject's `<surface>_sphere` registered to its template (FreeSurfer's `sphere.reg`), left and right pieces as contiguous runs | stored with each surface, as `frames compute` and the group tree read them; for a subject whose template differs from the fibers' template, a `registration` package (`docs/specs/2026-11-16-subject-centred-datasets.md`, section 4.4) |

`--fibers`, `--end-surface` and `--over` are required because each is a choice of data (decided 2026-11-16). A command that omits one refuses and names
it and its valid values.

**Type checks before anything runs** (preparation; each refusal names its reason): the fibers and the surfaces are in the same coordinate system
(SCS, metres); `nP ≥ 2`; the partition is a partition of THIS end surface; each registration sphere has as many vertices as its surface and the same
pieces, in the same order, as the other; the fibers belong to the subject or to its template (fibers of any other subject are refused); with template
fibers, the end surface exists under the same name on both.

**Options** (the options struct uses snake_case, the command line kebab-case, the provenance both):

| option | default | meaning |
|---|---|---|
| `max_end_distance` | `5` (mm; `none` = no gate, the reference's behaviour) | a fiber with an end farther than this from the end surface is left out (section 5.2) |
| `smooth_hops` | `3` (the reference's; `0` = none) | the hops of the mesh smoothing, applied to the partition's memberships (section 4.2) |
| `prune_below` | `1e-4` (the reference's; `0` = none) | **materialised connectome only:** connection weights below this fraction of the strongest are removed |
| `keep_largest_component` | on (the reference's) | **materialised connectome only:** also stores the largest connected component's vertices |
| `store` | off | with `--over vertices`: materialise and store the vertex connectome |

`prune_below`, `keep_largest_component` and `store` are refused with any partition other than `vertices`.

## 4. The graph

Every node is a pure function of matrices in `compute/` (matrices in, matrices out; callable with arrays made anywhere; no file, no subject, no
knowledge of where the fibers came from). Preparation reads the inputs and type-checks them; the graph runs once (there is no recording, so no chunk
loop); readback reduces and hands the outputs to io.

    fibers [nF·nP × 3] ──fiberEnds──▶ end points [2nF × 3]
        ──nearestVertex(end surface)──▶ end vertices [2nF] and end distances [2nF]              (the pointwise object, stored)
        (template fibers)  ──carryEnds(registration match)──▶ end vertices on the subject [2nF]

    readback (a measurement over the partition, `--over`):
        memberships [V × R] (one-hot) ──smoothMemberships(hops)──▶ smoothed memberships [V × R]
        end vertices, gate ──fiberPairs──▶ per fiber: smoothed membership rows of its two ends
        measure fiber count over the region × region partition ──▶ result [R × R]

    materialised, on request (`--over vertices --store`):
        end vertices ──endConnectome──▶ fiber counts [V × V] ──smoothingOperator(hops)──▶ connection weights
        ──pruneBelow──▶ ──componentsOf──▶ labels ──measure count──▶ sizes ──largest──▶ vertices kept

### 4.1 The ends (`fiberEnds`, `nearestVertex`)

- `fiberEnds(points, nP)`: the first and the last point of each fiber, [2nF × 3] (rows f and nF + f). Pure indexing.
- `nearestVertex(positions, query)`: for each query point the nearest vertex (Euclidean; the lowest index on a tie) and its distance. Exists as
  `nxr::crossings::nearestVertex` (brute force); moved to a general home (`compute/include/nxr/nearest.h`) and given a k-d tree (Bentley 1975) for
  10⁵ fibers on 10⁴–10⁵ vertices; same answers, tested against the brute force.
- The pointwise object is stored per bundle and end surface: int32 end vertices [2nF] (fiber f's ends at 2f and 2f + 1) and the distance of each end
  to the end surface (section 6). It does not depend on the gate, so one stored pair serves every gate and partition.

### 4.2 Counting over a partition, with smoothing

Raw ends leave most of the cortex without an edge: tractography stops at the grey/white boundary, so ends cluster on gyral crowns and miss the sulcal
banks. The reference bridges this by smoothing the vertex matrix over the mesh. Let the smoothing operator be the average of a vertex with its
neighbours (each row: the vertex and its neighbours, equal weights summing to 1; the random-walk normalisation of the mesh's adjacency, written from its
definition), applied `smooth_hops` times. In words:

    connection weight between vertex u and vertex v  =  sum over fibers of
        (smoothing of the first end onto u) x (smoothing of the second end onto v)  +  the same with the two ends swapped.

Counting over a partition sums those weights over the vertices of each pair of regions. Because the sum is linear, the smoothing moves onto the
memberships (decided 2026-11-16):

    smoothed membership of a vertex in a region  =  the smoothed one-hot membership: how much of the region its neighbourhood reaches
    result for regions a and b  =  sum over fibers of
        (smoothed membership of the first end in a) x (smoothed membership of the second end in b)  +  the same with the ends swapped,   a != b

so the vertex matrix is never formed. With `smooth_hops = 0` the memberships are the one-hot ones and the result is the plain fiber count between
regions. It is one sparse product per side: take, for each fiber, the smoothed membership row of its first end and of its second end ([nF × R] each,
sparse), and multiply one by the transpose of the other, then add its transpose. The cost is bounded by the number of fibers times the support of a
row, never by V². Between two different regions no vertex pair is a self-pair, so removing the vertex diagonal (as the reference does) changes nothing
here; it is a fact of the materialised path only.

This is a **measurement over a partition**, the `fiber_count` measure (its smoothed form `connection_weight`), reduced at readback by the measurement
functions: the fibers are the elements, the region pairs the members. It is not a node's own reduction (the cardinal rule). A smoothed membership is a
fractional membership; the measure sums the weights. Units: fibers (fractional when smoothed).

**Tiles.** `--over` a tile level of the dyadic ladder stores the result at the finest tile level; coarser levels are rolled up at read as sums of
2ˢ × 2ˢ blocks, as the measurement pyramid does, never recomputed and never stored per level. A region partition that is not a tile level is counted
directly.

### 4.3 The materialised vertex connectome (`endConnectome`, `smoothingOperator`, `pruneBelow`, `componentsOf`, `largest`)

On request (`--over vertices --store`), the vertex connectome is formed:
- `endConnectome(ends)`: the sparse symmetric matrix of fiber counts (one entry for each fiber, at its two end vertices, in both orders); the diagonal
  dropped (a fiber with both ends on one vertex leaves no entry). Eigen sparse, duplicates summed.
- `smoothingOperator(faces, V)` as above, raised to the power `smooth_hops` by sparse products (0: the identity); `sparseCongruence(P, A)` = P A Pᵀ
  with P and A sparse, a general-algebra function beside `nxr::algebra::congruence`. Then the diagonal is removed: the connection weights.
- `pruneBelow(weights, t)`, t = `prune_below` times the strongest connection weight (a `max` measurement read back first; with no entry nothing is
  removed). Entries below t are dropped from the pattern.
- `componentsOf`: the connected components of the graph whose edges are the non-zero connection weights, over all V vertices (an isolated vertex is its
  own component): `nxr::regions::detectRegions` with one value per vertex, numbered by their lowest vertex. The sizes are a `count` measurement over that
  labelling; `largest` takes the biggest (the lowest-numbered on a tie). The connection weights themselves are not restricted to it.

**Memory** (stated in the options struct). The smoothing operator holds about 37 non-zeros a row at 3 hops on a regular mesh; the smoothed matrix up to
nF times the square of that before pruning (10⁵ fibers: about 10⁸ non-zeros, about 1.6 GB in float64). The congruence is formed in column blocks and
pruned before the next is formed; the pruning threshold needs the strongest weight first, so there are two passes (the same result as one product).
This cost belongs to `--store` only: a partition result never forms the matrix.

## 5. Template fibers, the gate, and precision

### 5.1 Placing template ends on the subject

The template's ends are first placed on the template's end surface (`nearestVertex`), then carried to the subject through the registration of the
spheres, piece by piece (left to left, right to right):

1. `shepardWeights(destination sphere, source sphere, pieces, K)`: a sparse interpolation [V_subject × V_template]. For each subject vertex, its
   K = 8 nearest template vertices on the registered sphere (unit-normalised, so the weights are scale-free), squared chord distances d₁ ≤ … ≤ d_K, and
   the modified Shepard weights of Franke & Nielson (1980): weight k = ((d_K − d_k) / (d_K · d_k))², for k = 1 … K − 1, normalised to sum 1 (the K-th
   neighbour sets the radius and takes no weight; a zero distance is replaced by the smallest positive number of the type; a vertex whose weights are
   all 0 gives its nearest neighbour weight 1). Written from the paper's formula with these choices stated.
2. `strongestDestination(weights)`: for each template vertex, the subject vertex with the largest weight in its column (the lowest index on a tie).
3. **Template vertices with no subject neighbour** (an empty column: it happens whenever the template is denser than the subject) go to **their nearest
   subject vertex on the registration sphere** (decided 2026-11-16), a `nearestVertex` query on the sphere, piece by piece. The reference sends them to
   subject vertex 0; that is a defect of the reference, not a behaviour kept as an option.
4. `carryEnds(template end vertices, match)`: the subject vertex of each end.

A subject's own fibers take `nearestVertex` on the end surface directly; no registration is made. The same holds when the fibers' surface IS the end
surface (computing on the template itself).

### 5.2 The distance gate

`--max-end-distance` (mm; default **5**, the distance the existing template import uses; decided 2026-11-16):
- a fiber with an end farther than this from the end surface is **left out**; the number left out, of nF, is reported in the command's output and
  stored in the provenance (`n_fibers_left_out`, with the gate used);
- **each end's distance is stored either way** (section 4.1), so the gate can be changed, or the fibers left out looked at, without recomputing the ends;
- `none` applies no gate and reproduces the reference, where every end takes its nearest vertex however far it is.

For template fibers the distance is that of the end to the template's end surface, before the registration; the registration's own displacement is not
gated. The gate is part of the cache key.

### 5.3 What the instrument cannot resolve

Written into the provenance's `warnings` and the method's section in `docs/reference.md`: tractography ends are not the fibers' cortical terminations
(streamlines stop at the grey/white boundary); an end's vertex is a location on the chosen surface, not a depth; smoothing spreads ends over the mesh
and is not conserving (the rows of the smoothing operator sum to 1, its columns do not on an irregular mesh); a template's fibers on a subject's cortex
carry the template's anatomy through a spherical correspondence, not the subject's own white matter.

### 5.4 Precision

Precision follows the input (decided 2026-11-16): fibers in float32 give end distances and results in float32; float64 give float64. The smoothing
operator, the memberships and the Shepard weights are built in the precision of the surface positions they come from. No node casts silently; where two
inputs differ in precision, preparation refuses and names both, unless the caller converts one explicitly. The cache key includes the precision.

## 6. What is stored

One storage form (decided 2026-11-16): the schema `io/schema/nxr.connectome.schema.json` (a connectome under `manifold/` with a `keep` array) never
had a writer and is retired. Names and kinds are those the importer writes for the same thing, so a reader tells a native result from an imported one
only by its producer.

- **The pointwise object**, per bundle and end surface: `<fibers>_ends_<surface>`, int32 [2nF], and `<fibers>_end_distance_<surface>`, the distance of
  each end to the surface, in metres, in the precision of the fibers.
- **A partition result** (`--over` a partition or a tile level): `fiber_count` measurement rows on the partition Selection, two codes a row (`code`, the
  region; `within_code`, the other), both orders, value the count (`connection_weight` when `smooth_hops > 0`). For tiles, only the finest level is stored
  and coarser ones are rolled up at read. The rows belong to a byte-less Field `<name>` of kind `connectome` that carries the provenance (the precedent of
  the tile-map fields: rows of a computed measure point to a Field).
- **The vertex connectome** (`--over vertices --store` only): a Field of kind `connectome`, SPARSE (CSC) on the product manifold surface × surface at
  `source-source/<name>`, unit `fibers`, in the input's precision; and `<name>_component`, a Selection of the largest component's vertices naming the
  connectome field in its `params_json`. Without `--store`, `--over vertices` is refused with a message that says so (the vertex matrix is large).
- `<name>` defaults to `<fibers>_<surface>_fiber_counts` (with `smooth_hops = 0`) or `<fibers>_<surface>_connection_weights_h<hops>`, with `_on_<surface>`
  for template fibers; `--name` sets it. The importer's own unsmoothed `<fibers>_connectome` is never overwritten.

**Provenance** (`producer_json`, and every option in the cache key): `method` (`"fiber end counts over a partition"`), `reference`
(`"tess_connectome (MATLAB)"`), `over`, `end_surface`, `max_end_distance`, `smooth_hops`, `prune_below`, `keep_largest_component`, `registration` (`none`
or the registration used, with its name, version and content hash), `fibers` (the bundle's subject, name and id), `n_fibers`, `n_fibers_left_out`,
`precision`, and, for template fibers, the bundle's `origin` and licence (section 7). A rerun with the same inputs and options is a cache hit;
`--replace` recomputes. The warnings of section 5.3 are written.

## 7. Template fibers: HCP-1065

The template package carries **HCP-1065**, the population-average tractography of DSI Studio (Yeh 2022, *Nat Commun* 13:4933), imported natively by
`import template` (decided 2026-11-16) with its full `origin` record: source, version, citation, and licence **CC BY-SA 4.0 together with the WU-Minn
HCP Data Use Terms**.
- A derived connectome carries the bundle's licence and attribution in its provenance (ShareAlike: a result derived from it is shared under the same
  terms); preparation copies the `origin` into the result's record, and `template release` already refuses an item without one.
- A subject without its own fibers names the template's bundle (`--fibers <template>:<fibers>`) and uses it through the registration between the
  template and the subject (section 3). A subject that has its own fibers uses them by naming them: nxr-neuro never falls back from one to the other.
- `--end-surface` is the user's choice and is read on the template and on the subject by the same name. The pial surface is the one source maps are
  written on, and is the natural choice for comparing with them; the assignment is a location on the cortex, not a claim about the depth at which a
  fiber ends.
- Which edition of the template carries it, full or redistributable, follows its `origin` licence: the licence permits redistribution with attribution
  and ShareAlike, so both editions hold it.

## 8. The command line

Verb first (`docs/specs/2026-11-16-command-line.md`); the help of this command is per command, from its options struct's comments (meaning, default,
unit), with the examples below.

    nxr-neuro compute connectome --fibers <bundle> --end-surface <surface> --over <partition>
                                 [--smooth-hops N] [--max-end-distance MM|none]
                                 [--name NAME] [--replace] --dataset D --subject S

Materialised vertex connectome (the pruning options only here):

    nxr-neuro compute connectome --fibers tracts --end-surface pial --over vertices --store \
                                 [--prune-below 1e-4] [--no-keep-largest-component]

Examples:

    # fiber counts between the regions of a parcellation, no smoothing, own fibers, no gate (the reference's ends)
    nxr-neuro compute connectome --fibers tracts --end-surface pial --over desikan-killiany \
                                 --smooth-hops 0 --max-end-distance none --dataset d --subject s
    # a subject without fibers: the template's, on the pial surface, tiles at the finest ladder level
    nxr-neuro compute connectome --fibers hcp1065:tracts --end-surface pial --over tiles:7 --dataset d --subject s

- `--fibers`, `--end-surface` and `--over` are required; `--prune-below`, `--keep-largest-component` and `--store` are refused with any other `--over`
  than `vertices`.
- Output: JSON lines (progress: `ends`, `registration`, `counting`; then `"stage": "result"` with the names written, the number of fibers left out by
  the gate and `cache`).
- Until the verb-first grammar lands, the same verb is also `connectome compute`; one parser, one dispatch.

**How results differ from the reference.** (1) A fiber end belongs to the region of its vertex; the reference assigned it to the region whose
centroid was nearest. The two differ wherever an end is nearer another region's centre than its own region's (long or curved regions): region results
differ from the reference's there, and this is intended. (2) Unmatched template vertices go to their nearest subject vertex, not to vertex 0. (3) The
gate defaults to 5 mm; `none` reproduces the reference's ends.

Files: `compute/include/nxr/connectome.h` (+ `nearest.h`, `shepard.h`, the sparse congruence in `congruence.h`),
`pipelines/include/nxr/pipelines/connectome.h` (`ConnectomeOptions`, `connectome_compute(ds, sub, opt, progress)`), `cli/connectome_command.cpp`, and the
method's section in `docs/reference.md`.

### Checked against the Studio's chain

The Studio's `connectome.ts` has the same pieces, ported with its names: the bundle's `ends` (int32, two a curve, −1 off the surface), the vertex
connectome as a pass over the curves, `tileConnectome` (a partition's matrix times the vertex matrix times its transpose), `rollUp` (a coarser ladder
level as a sum of 2ˢ × 2ˢ blocks) and `frameConnectome`. Counting over a partition with smoothed memberships is `frameConnectome` with the smoothing as
its atoms, applied to the ends rather than to the formed matrix. The gate, the pruning and the components are new nodes the Studio does not have; no
stage needs the Studio to change.

## 9. Tests

Synthetic data only (spheres, lattices, generated fibers); nothing from a real dataset.

**Unit tests, one a node** (`compute/test/test_connectome.cpp`):
- `fiberEnds`: the first and last rows, order f then nF + f.
- `nearestVertex` (k-d tree): equal to the brute force on random clouds; the lowest index on exact ties; the distances.
- `endConnectome`: symmetric; the total is 2 times the fibers with distinct end vertices; a self-pair leaves no entry; two fibers on one pair give 2.
- `smoothingOperator`: rows sum to 1; 0 hops gives the identity; on a regular lattice the support after p hops is the p-ring.
- `sparseCongruence`: equals the dense P A Pᵀ on small random sparse matrices.
- `pruneBelow`, `componentsOf` + `largest`: by hand on a 4 × 4; two blocks of different sizes; a tie goes to the lowest; isolated vertices are components.
- `shepardWeights`: rows sum to 1; a destination on a source point gives weight 1; the K-th neighbour has weight 0; invariant to the sphere's radius;
  the pieces never mix.
- `strongestDestination`: ties to the lowest.

**Golden tests against the reference** (decided 2026-11-16; `compute/test/golden/connectome_golden.txt`, its generator `gen_connectome_golden.m` beside
it). The generator is run ONCE by a developer in MATLAB with the reference's Brainstorm environment, never at run time and never by a test. It builds a
throwaway protocol of SYNTHETIC data: two icospheres (left and right pieces), their spheres, and bundles of straight fibers with a random bend, some
within a hemisphere, some across, a few with both ends on one vertex and a few duplicated, with ends drawn clear of ties between two vertices. It writes
its inputs and outputs as text (sorted triplets to 17 significant digits).
- **The vertex-level fiber counts, with the gate `none`, on quirk-free cases** (own fibers; template fibers where no template vertex has an empty column,
  i.e. the template is not denser than the subject) are compared with the reference's: identical in pattern and value. The reference's smoothed
  connection weights, pruning and largest component are compared at the same cases (0, 1 and 3 hops; pruning 0 and 1e-4): pattern identical, values to
  1e-12 relative (the order of sparse sums differs from MATLAB's).
- The registration map on those cases (template vertex to subject vertex) is identical.
- Cases with empty columns are not in the golden: the reference's result there is the quirk, which is fixed.

**Region results** (`test_connectome_partition`): equal to **a direct sum over the partition** of the vertex fiber counts (and, smoothed, of the vertex
connection weights, formed in the test only) for random partitions, a tile level and a hemisphere pair; equal at 0, 1 and 3 hops to 1e-12; rolling the
finest tile level up to a coarser one equals counting over the coarser level directly; the result does not depend on the order of the fibers.

**The gate and the quirk fix, each its own test** (`test_connectome_gate`, `test_connectome_registration`):
- the gate: fibers with an end farther than the gate are left out and counted in the reported number and the provenance; their distances are stored;
  `none` leaves none out; a changed gate is a cache miss and reuses the stored ends; a gate between two distances separates exactly those fibers.
- the quirk: with a template denser than the subject, every template vertex of an empty column sends its ends to its nearest subject vertex on the
  sphere (checked against a brute-force search), and no end goes to vertex 0 unless that is the nearest.

**Pipeline test** (`pipelines/test/test_connectome_pipeline.cpp`, registered in `pipelines/CMakeLists.txt`): a synthetic dataset with a subject, its
template, surfaces, spheres, a partition and fibers written through io. Own and template fibers; every `--over` kind; a stored vertex connectome with
`--smooth-hops 0 --prune-below 0` equals the importer's `<fibers>_connectome` minus its diagonal; precision follows the input (float32 in, float32
out); the template bundle's licence and attribution are in the result's provenance; a rerun is a cache hit and a changed option a miss; refusals
(`--fibers`, `--end-surface` or `--over` missing; template fibers without a registration; a sphere of the wrong size or pieces; a partition of another
surface; `--prune-below` with a partition; `--over vertices` without `--store`; fibers of an unrelated subject; mismatched coordinate system or precision).

Tests run through `tools/tests.py run connectome`, scoped to these.

## 10. Open questions for the maintainers

1. **Within-region counts.** The reference leaves out fibers with both ends in one region, and the partition result keeps that (rows for a ≠ b only).
   Store the within-region count too (a row with `code` = `within_code`)? It is cheap and says how much of a region's fibers stay inside it.
2. **The default of `--smooth-hops` for partitions.** The reference smooths over 3 hops, and the default here follows it so that the vertex golden
   holds. For counts over regions, 0 (plain fiber counts) may be the better default; keep 3, or default to 0 for partitions?
3. **Gate and registration.** The gate measures an end's distance to the template's end surface, before the registration, and does not gate the
   registration's own displacement. Is a second gate, on the distance between a template vertex and the subject vertex it is carried to on the sphere,
   wanted?
