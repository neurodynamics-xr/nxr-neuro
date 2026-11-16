# The dataset's default subject in nxr-neuro, step 1: the FreeSurfer surfaces, the parcellations, the frames, the group tree and the precomputed modes (2026-11-11)

## Goal

nxr-neuro implements the default subject as part of its dataset functions, starting with the basics: the FreeSurfer surface and its pre-computed eigenmodes, prepared as a regular subject is.

Every subject of a dataset is registered to one average anatomy (Brainstorm's `@default_subject`; nxr's D127). Neither nxr nor nxr-neuro made it
when a dataset was created: nxr builds it afterwards (`nxr convert atlas default-subject`), nxr-neuro had no builder.

## What is built

`nxr-neuro dataset create --datastore D --name N --templates <FreeSurfer subjects folder> [--default fsaverage5] [--modes K] [--atlas-depth 8] [--lc-levels 4,6]`

- **The subject:** `@default_subject`, kind `template`, store `@default_subject.nxr.zarr`, source FreeSurfer. Written as any subject is: `SubjectWriter`, `add_surface`,
  the layout rule, one composition. Subjects created later point at it (`template_id`).
- **The surfaces:** `cortex_pial` (primary), `cortex_white`, `cortex_inflated` from `<templates>/<default>/surf/{lh,rh}.*`.
  - Left then right, mm converted to m.
  - Hemispheres `_1` (left) and `_2` (right).
  - `cortex_pial_sphere` is `sphere.reg` as stored (radius 100, no unit change), as nxr's default subject writes it.
  - Coincident vertices are separated (nxr's `separate_coincident`; fsaverage ico7 has one, fsaverage5 none).
  - Names follow nxr's default subject.
- **The modes:** with `--modes K` (pipelines built), `dataset create` solves the pial's Hodge–Dirac basis with `eigen solve`'s pipeline (`nxr::pipelines::eigen_solve`).
  - The solve is compute's, so it is run by `nxr::pipelines::create_dataset_with_template` (the CLI only parses and prints), never by io (`default_subject.cpp` writes no modes).
  - Without `--modes`, `eigen solve` on the default subject gives the same basis. It is stored as `eigen solve` stores any subject's, glued surface and bridge included.
  - Modes are read and written 64 at a time, so peak memory is about 26 MB for fsaverage5.
- **The parcellations** (added the same day: "ok, let's add the parcellations"): the pial's selection set `source/cortex_pial_atlases`, exactly nxr's
  `DefaultSubjectWriter.parcellations`, written by `add_surface` (the regular subjects' writer of Brainstorm's parcellations):
  - **Members:** Structures (1 left, 2 right), Desikan-Killiany (`label/{lh,rh}.aparc.annot`), Destrieux (`aparc.a2009s`).
  - **Codes:** each member is a partition, one int32 code per vertex, coded by an app-scope dictionary of the family's name. Code 0 is `unassigned`; the
    regions are "<region> L|R" with FreeSurfer's colours.
  - **Left out:** `unknown`, `Medial_wall`, `corpuscallosum` and regions with no vertex stay unassigned. A family missing either hemisphere's file is
    left out whole.
  - **Reader:** `read_freesurfer_annot` (`surface.h`) is nibabel's `read_annot`, its two quirks included: new-format names in reading order, labels
    placed by `searchsorted`.
- **The group tree**: the dyadic ladder of the default pial,
  the same `ladder` Selection (D85) `tiles compute` writes (`nxr::pipelines::group_tree_compute`, called by `dataset create` after the subject):
  - **Form:** `cortex_pial tiles`, type `ladder`, members `cortex_pial tiles L0 … L<finest>`, each a partition with one tile code `h·2^l + t` per vertex and
    its own tile dictionary. nxr writes its group tree as a plain `set` (`source/cortex_pial_tiles`), not D85's `ladder` type; this follows the schema and the Studio, so nxr's atlas code will need to read the ladder.
  - **Marked as the group's:** `tree: "group"`, and the dataset's default atlas definitions in the ladder's `params_json.atlas`, with nxr's defaults:
    leaf depth 8 (`--atlas-depth`), the resolution, frames null (not yet written), the tower time convention (level −18, ≈ 0.33 s), the default bands
    (octaves 1–64 Hz), the Levi-Civita levels 4 and 6, the trajectory scales.
  - **A bug found on the way:** `tiles compute` read positions rounded to float32 (`read_surface`'s default, meant for the GPU-side pipelines), while
    the Studio's stored ladder (backend `loadPositions`, float64) and nxr cut the float64 positions. On fsaverage5 a few vertices moved from level 7 down
    (169 at level 11). `read_surface(…, as_stored = true)` now gives the ladder and the tile summaries the stored float64.
- **The frames**: `frames compute` on the default pial,
  called by `dataset create` before the group tree.
  - **Two separate things.** The optical flow is computed with the connection Laplacian (gauge-free: rotating every frame leaves the velocity unchanged; the default since
    2026-11-11, Rheome's frame-coordinate smoothness removed). The frames only READ directions out, after the solve. Rheome's problem was the first kind: its smoothness
    aᵀKa + bᵀKb depended on the (1,1,1)×n frames (a 180 % change in velocity when their axis was turned).
  - **Canonical frames:** `cortex_pial_canonical_frames` ([V,3,3]: north, west = n × north, n), `cortex_pial_frame_singularities` (the two poles a hemisphere),
    `cortex_pial_canonical_gauge_angle`. North is the meridian of the registered sphere towards +z, pushed forward to the pial (Rheome's `sphereframe`, closed form; nxr
    solves the same field with a trivial connection). Read from the positions as stored (float64), as nxr and Rheome do; `frames compute` now does this for every subject.
  - **Anatomical frame** from the default subject's own axes, FreeSurfer's surface RAS (`--convention ras`): the template every subject's anatomical frame is carried from.
    nxr's default subject has none; this is nxr-neuro's addition (`frames compute`, 2026-11-11).
  - The group tree's definitions name the canonical frames (`atlas.frames`, the field's id), as nxr's do.
- **The Levi-Civita connection of the group tree** (nxr's default subject, `measure.vectors.levi_civita`), at `--lc-levels` (4, 6):
  - `cortex_pial_lc_centres_L<l>`: the centre of each tile (the non-singular member nearest the area-weighted centroid; nxr's `node_centres`).
  - Ω[a, b]: the angle at centre b (from north towards west) of centre a's north carried there by the vector heat method, a `levi-civita connection` measurement (rad) on level l's
    Selection, code a, within_code b (D158). None across hemispheres. The record (transport, antisymmetrisation, warnings) is in the ladder's params, `levi_civita`.
  - **A bug in nxr, not ported:** nxr antisymmetrises with `angle(e^{iΩab} e^{−iΩba}) / 2`. Halving an angle on the circle is ambiguous by π: every pair with |Ω| > 90° comes
    out turned by 180°. Here: Ω = Ωab − wrap(Ωab + Ωba)/2 (half the round-trip defect taken from each direction). nxr's stored default subjects carry the flip on about a quarter
    of their pairs (fsaverage5: 142 of 512 at level 4, 2 140 of 8 192 at level 6). nxr needs the same fix.
  - **Transport:** the Studio's (`ChartKit::transported`, intrinsic Delaunay); nxr's is geometry-central's on the input mesh.
- **Both or neither:** a failure in the default subject removes the dataset the command created.

New code in `io/` (no compute dependency): `npy.h` (the .npy reader), `read_freesurfer_surface` (`surface.h`; `read_mesh` reads `lh.*`/`rh.*` with it,
and `bst::read_hemis` is implemented on it), `default_subject.h`. The command: `cli/nxr_neuro.cpp` (`dataset create`).

## Why the modes are imported, not solved

`eigen solve` refuses a cortex of two pieces: the Hodge–Dirac solve needs the hemispheres glued by anatomy (the Desikan–Killiany medial wall cut, a bridge
grown: the Studio's glue), which is not ported. So the package is a solve made elsewhere:
- **Source:** `brain-maps/modes/derivatives/nxr-modes-hodge-dirac/pial_fsaverage5` (nxr-convert 0.2.14 `modes`).
- **Mesh:** glued, 19 152 vertices, 18 426 of fsaverage5's 20 484 kept, a bridge of 726.
- **Modes:** 2 944 (1 501 exact, 1 443 coexact).
- **Unrecorded options:** its `hodge.json` records no ⋆1 floor, shift, tolerance or mollification. Whether it is the reference method of 2026-11-10 is not
  known from its files; the provenance stored says only what the files say.

## Validation

- **C++** (`io/test/test_default_subject.cpp`, synthetic):
  - The FreeSurfer and .npy readers.
  - Every row and array of the default subject.
  - Refusals: a second default subject; a templates folder without `surf/` (nothing is left).
- **Modes** (`pipelines/test/test_pipelines.cpp`, synthetic icosphere): `eigen_solve` on a template subject's `cortex_pial` stores the basis in the layout of any subject's (`hodgeDirac_eigenvectors`, `hodgeDirac_forms2`, the glued manifold).
- **Parity** (`io/test/parity/test_default_subject_parity.py`, synthetic, nxr's environment):
  - `dataset create --templates` surfaces equal nibabel's `read_geometry` / 1000.
- **Parcellations, parity** (`test_d_the_parcellations_are_nxrs`): nxr's `DefaultSubjectWriter.parcellations` and nxr-neuro, run on the same synthetic
  annotations (with `unknown`, `Medial_wall` and an empty region), write equal rows, dictionaries (names, codes, colours, order) and codes. Proved
  able to fail: with `Medial_wall` not excluded it fails.
- **Parcellations, real data:** fsaverage5 from `brain-maps` against the `@default_subject` nxr built (fsaverage5) in a Cortical Flow datastore dataset: every one of
  20 484 vertices has the same region in all three families, with the same codes, colours (69 Desikan-Killiany and 149 Destrieux levels) and set description.
- **Group tree, parity** (`test_e_the_group_tree_is_nxrs_ladder`, bumpy icospheres of 642 vertices a hemisphere): every level's codes equal nxr's
  `surface_ladder` per hemisphere; the ladder row, `tree: "group"` and the atlas definitions equal nxr's constants. Proved able to fail: with the float32
  positions it fails (level 7, 5 vertices).
- **Group tree, real data:** on fsaverage5, levels 0–11, the codes equal the Studio's own `surfaceLadder` (run through its TypeScript) and nxr's, vertex for
  vertex. The group tree stored in an older nxr-built `@default_subject` differs from both from level 8 down (an older ladder).
- **Frames and Levi-Civita, parity** (`test_f_the_frames_and_the_levi_civita_connection`, two exact spheres of 642 vertices, where both are known in closed form):
  - north within 1° of the meridian, n the sphere's normal, e2 = n × e1, the four poles singular;
  - the tile centres equal nxr's `node_centres`;
  - Ω has a value for every pair in a hemisphere and none across, Ω[b, a] = −Ω[a, b], and it equals the exact transport (the rotation about a × b along the great circle) to a median
    under 0.15° and a maximum under 1° (pairs under 120° apart). Measured: median 0.04° (L4) / 0.06° (L6). nxr's transport on the same spheres: median 0.06°, max 2.9°.
  - Proved able to fail: with nxr's antisymmetrisation, all 120 pairs with |Ω| > 90° at level 4 are off by 180°.
- **Frames and Levi-Civita, real data** (fsaverage5, against the default subject nxr built in a Cortical Flow datastore dataset, read only; that dataset stores fsaverage rotated about
  z, which was undone first):
  - canonical frames: the same four singular vertices; north within 0.003° (median), 0.16° (p99), 1.3° (max) of nxr's;
  - tile centres equal at levels 4 and 6;
  - Ω, modulo nxr's 180° flip: median 2.6° (L4) / 1.9° (L6), p90 12° / 8°, max 67° / 86°, growing with the distance between centres (median 1.1° under 20 mm). The two meshes'
    transports differ on folded cortex between distant tiles; on the sphere both are within 0.1°. This is a limit of the measurement, recorded in its warnings.
## Not yet (nxr's default subject has them)

- The option resolutions (fsaverage6, fsaverage) with their restriction operators.
- HCP-1065.
- The Frequency manifold.
