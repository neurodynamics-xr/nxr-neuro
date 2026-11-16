# Measurements of Fields over Selections (spec, 2026-11-11; brought up to date 2026-11-12 with what is built and decided)

**Status:**
- **Scalar Fields: built and merged** (nxr-neuro #60, #62, #63; Studio #184; nxr #78). §3–4.
- **Vector Fields: built and merged** (nxr-neuro #64, #65; nxr #79; the Studio's schema copy, #191, is open). §5.
- **The sensor cross-spectrum and the first pair quantity (the phase gradient): built and merged** (nxr-neuro #66, #67). §6.
- **How recordings are read (the chunk service): built and merged** (nxr-neuro #68, #69, #70). §7.
- **The decisions taken**, each dated, are in §8. **What remains**, with the decisions still open, is in §9.

## 1. Why

A measurement in this lab is made in **two stages**, and both are stored:

1. **The result Field.** It is vertex-resolved: an average over time or frequency of a recording's sources, such as a band power map. It carries
   its provenance and is what is drawn and compared with a reference. A one-time comparison on 2026-11-11 (OMEGA sub-0181, not kept as an
   audit) matched Brainstorm's band power maps at this stage to float32. The end-to-end proof of dataset creation and import from Brainstorm is the
   a real-data fixture kept outside the repository (`NXR_FIXTURE_DATASET`).
2. **The measurement.** The Field is measured over the members of a partition (Desikan-Killiany regions, a level of the tile ladder), giving one
   value per member and measure.

Statistics across people are run on the second stage, so it must be exact, mergeable (it rolls up to coarser members and pools across subjects)
and recorded (how each number was made).

## 2. One system: partition Selections and measurement functions

There is no separate "reduction" library and no vocabulary beside the data model's:

- **The partition is a Selection** (D48): `type = 'set'` with a code per element and a dictionary that names the codes. Desikan-Killiany and the other
  imported parcellations are partitions. Each level of a tile ladder is a partition (D85), so the ladder's tiles are measured the same way.
- **The measurement functions are the Studio's** (`backend/src/db/measurements.ts` `measureOverPartition` and the measure registry,
  `io/schema/measures.json`). The Studio is the reference; nxr-neuro ports them (`compute/include/nxr/partition.h`). The Studio's live views (Tiles
  scope, Measure on a Mask, channel groups) compute the same quantities over the same partitions and are not separate methods.
- **The rows are the Studio's** (`selection_measurement`, D68): selection = the partition, code = the member, of_field_id = the Field, with the
  member's `weight`.
- **The lookup is the Studio's** (`query_measurements`): roll-up by `code >> shift`, and derived measures (variance, sd, rms) computed at read from
  their stored ingredients.
- **The record of how a measurement was made is a recipe** (the Studio's recipes, `docs/recipes.md`): an entry of the dataset's `recipes`
  dictionary, named in every row's `computed_by`.

## 3. Scalar Fields (built)

`nxr-neuro measure compute --dataset D --subject S --labelling L --fields PATTERN --weight none|barycentric|voronoi [--measures …] [--recipe NAME] [--replace]`

- **Measures, per member and Field:**
  - `mean` = Σw·x / Σw and `mean_square` = Σw·x² / Σw, each with `weight` = Σw;
  - `max`, `min`, `abs_max` and, on request, `sum`.
- **Measures, per member once:** `n_elements`, and `area` (m²) when the weights are areas.
- **Merging is exact.** A stored mean with its weight is its sum (mean × weight = Σw·x), so a mean rolls up to a coarser member, and a variance is
  derived from the rolled-up mean and mean_square.
- **The weight is an option with no default:**

| `--weight` | Per vertex | Whose |
|---|---|---|
| `none` | 1 | Brainstorm's scout mean: the plain vertex mean (checked against Brainstorm's own scout function: identical on all 68 Desikan-Killiany regions) |
| `barycentric` | a third of each incident triangle | nxr's atlas, Rheome, the Studio's tile ladder |
| `voronoi` | the mixed Voronoi area of the mollified input | the Studio's `lumpedMass`, nxr-neuro's operators |

  On OMEGA sub-0181's alpha map, the plain and area-weighted means differ by 14 % in the median region and 31 % at most.

- **The recipe** is `partition-<weight>@<version>`, kind `reduction`, carrying the method, weight and measures. A new version is made when the
  choices change.
- **One recipe's rows at a time** per (partition, Field, measure): the table's key has no recipe.
  - A re-run of the same recipe is a cache hit.
  - Another recipe's rows are refused unless `--replace`.
- **Departures from the reference, each counted, never silent:**
  - A negative code is in no member.
  - A non-finite value makes its member unwritten.
  - A code with no vertex is written with a mean of 0, as the reference does.
- **Checks:**
  - The port equals the Studio's function to the last bit (`test_partition`; golden from the Studio's own code).
  - On OMEGA sub-0181, `--weight none` equals Brainstorm's scout means of Brainstorm's own maps to 1.3e-8.
  - The pipeline test checks each weight against an independent computation, and the cache, refusal, replace, recipe and derived sd.

## 4. The measurement table (built)

`nxr-neuro measure export --dataset D --labelling L --fields PATTERN --table NAME --out DIR [--statistics mean,sd,…] [--subjects …] [--recipe …] [--layout split|wide] [--exclude … | --keep-all]`

The layout is a study release's: tables plus a data dictionary, following nxr's group-analysis design §8.

- **Split by default (#62):** one table per condition × statistic, `<table>_<condition>_<statistic>.csv`.
  - A row is a subject and a column is a member (`meg_dk_alpha_mean.csv`, column `precuneus_L_mean`).
  - The members' own area and vertex count go in `<table>_<statistic>.csv`.
  - A cell is empty where a subject lacks the value.
- **`--layout wide`:** one `<table>.csv`, with a column per condition × member × statistic.
- **`dictionary.csv`:** one row per column of every table in the folder.
  - The release's columns: `index, data_table, column_number, variable, description, units, possible_data_values, value_type, notes`.
  - Then ours: `labelling, member, condition, statistic, field_kind, recipe`.
- **`<table>.json`:** the provenance: version, request, dataset, each recipe's choices, and per subject the partition and Fields read.

**Rules:**
- **Read only.** Every value goes through the lookup, so derived statistics come from the stored rows.
- **The condition** of a Field, which makes it the same measurement in every subject, is the first capture group of `--fields` (`'.*_psd_(.*)'`
  gives the band).
- **Members are joined across subjects by dictionary name.**
- **`unassigned` (the medial wall) is left out by default**. `--exclude` names others; `--keep-all` keeps every member.
- **Refused:**
  - two Fields of one condition in a subject (ambiguous);
  - one column made by different recipes in different subjects (one column is one method).

## 5. Vector Fields (built)

**The decision ( and 2026-11-12).** A vector Field is summarised over time the way a scalar is: band power is the time summary of a scalar
signal, and its vector analogue is the band's **cross-spectral tensor**. It is stored whole and the readable maps are derived from it:
- **Stored:** one complex 3×3 Hermitian matrix per vertex and band, T = ⟨V Vᴴ⟩ of the current's complex amplitude, in ambient components (gauge-free and
  complete). Its trace is the band power; its real part is where the current goes (axis, anisotropy), its imaginary part how it turns.
- **Derived:** the tangential Stokes quaternion [S0, S1, S2, S3] in the canonical frames, the normal power nᵀ Re(T) n, and the normal share.
- **Tangential fields** (optical flow velocity, gradients): the Stokes quaternion is the complete summary and is stored directly.
- **Why a tensor and why Hermitian**: the summary must be independent of phase
  and additive over time, vertices and subjects, so it is second order. The real covariance cannot tell clockwise from counter-clockwise; the quadrature
  (imaginary) part can. In xyz the summary needs no frame; frames come in only when the readouts are derived.

**The value type `hermitian3` (#64; nxr #79; the Studio #191, open).** float32 [.., 3, 3, 2], 18 components. It was added **without a schema version
bump**: an older database's CHECK lists are relaxed in place, once, when a `hermitian3` Field or
Operator is first written, and the change is noted in `meta`.

**The source tensor (`sourcetensor compute`, #65).** T = K_v (S + i Sq) K_vᵀ per vertex and band from the sensor cross-spectrum, through a
free-orientation kernel. Stored per band: `<stem>_<band>` (hermitian3), `_stokes_` (quaternion), `_normalpower_`, `_normalshare_`. Tested: the trace
equals `sourcepsd`'s band power exactly; the sign of S3 is the turning sense seen from the normal.

**The measurements (`measure compute`, #65).** The same system as the scalars (partition, measurement functions, rows, recipe, lookup):
- **Gauge: the canonical frames** (e1 north, e2 west, n out, from the registered sphere; `frames compute`). Built as this spec recommended (V1).
- **`tensor_*`** (registry codes 40–49): per member, the area-weighted sums of T's components in the frames (11, 22, nn, 12, 1n and 2n in real and
  imaginary parts, and the total over every vertex). nxr's `band_tensor` layout.
- **`gradient_*`** (codes 30–37, `--gradient`): the gradient of each scalar Field (nxr's `vertex_gradient`) as sums in the frames (nxr's `vector_stats`).
- **One recipe per family:** `partition-<weight>`, `partition-<weight>-gradient`, `partition-<weight>-tensor`, so a choice in one family never refuses
  another's rows. Size rows are shared when the weight is the same.
- **Region readouts are ratios of sums**, never means of per-vertex ratios. The export derives them (the gauge-free ones first, as V2 recommended):
  - tensor: `normal_share`, `anisotropy` |S1 + iS2| / S0, `circularity` S3 / S0, `power_density`; and `axis_deg`, which depends on the gauge;
  - gradient: `gradient_strength`, `gradient_coherence`; and `gradient_direction_deg`, which depends on the gauge.
- **Tested** on known Fields: the gradient of z points north (within 1°); the tensor z zᵀ against an independent computation (to 2e-16).
- **What a region cannot show.** Orientations turn with the folds, so a large member averages them away. On OMEGA sub-0181 (alpha) the anisotropy is
  0.57 at a vertex and 0.05–0.3 per Desikan-Killiany region; smaller members (tiles) keep more of it.

## 6. The sensor cross-spectrum and pair quantities (built)

**The decision.** The band's sensor cross-spectrum C is **stored** as the root of every source
cross-spectral quantity (`csd compute`, #66). Every block between two places is K_i C K_jᵀ, read on demand (`nxr::meeg::crossTensor`): the 3N × 3N
source cross-spectrum (about 16 GB a band) is never formed, and one C serves every kernel whose channels it covers. It is stored as the noise
covariance is: an Operator of kind `cross-spectral density`, value type `complex`, on its channels' Selection.

**The phase gradient (`phasegradient compute`, #67).** The first pair quantity: whether the band's activity moves across the cortex, which way and how
fast (m/s). nxr's phase reader (`nxr.detect.core_phase`, ∇φ = Im(z̄ ∇z)/|z|²) in spectral form, read from the stored C through the kernel and the mesh's
P1 gradient. Its sums (P, H, Q) add over regions, so a region's readout is a ratio of sums. Tested on ground truth: speed within 1.3 % (median),
direction within 1.4°, travelling index 1.000 for a travelling wave and 0.003 for a standing pattern.
- **What it is:** one power-weighted average over the cross-spectrum's span and band. It is **not time-localised**: waves that come and go in
  different directions cancel. It is not travelling-wave detection.

## 7. How recordings are read (built)

**The decisions.** A recording can be hours long and is never loaded whole.
- **A chunk** is 2^15 samples of the time axis, every channel. It has no other meaning.
- **The default is one chunk**, never the whole recording. The whole recording is an explicit loop over chunks (`--chunks all`).
- **A chunk is not the FFT window.** A Welch segment's length, taper and overlap are the computation's own choice and are carried across chunk edges.
- **Only io's chunk service reads a recording** (`io/include/nxr/io/chunk_loop.h`, the Studio's `ChunkSource` and `chunkLoop` ported). Pipelines
  declare what they read and fold each chunk; compute receives matrices. The test `layering_recordings_read_by_chunk_service` enforces it.

## 8. Decisions taken

| Date | Decision (the maintainers') | Where |
|---|---|---|
| 2026-11-11 | One system: partition Selections and the Studio's measurement functions; no parallel vocabulary ("tile means" and the like) | §2 |
| 2026-11-11 | The weight is an option with no default (`none`, `barycentric`, `voronoi`) | §3 |
| 2026-11-11 | The measurement table follows a study-release layout; `unassigned` left out by default | §4 |
| 2026-11-11 | Scalars first, then vectors | §3, §5 |
| 2026-11-11 | A vector Field's time summary: the full complex 3×3 tensor per vertex and band, the readable maps derived; tangential fields store the Stokes quaternion | §5 |
| 2026-11-12 | `hermitian3` without a schema version bump, older databases relaxed in place | §5 |
| 2026-11-12 | The band's sensor cross-spectrum is stored; pair blocks are read on demand | §6 |
| 2026-11-12 | One chunk by default; the chunk is not the FFT window; only io's chunk service reads recordings | §7 |

Built as this spec recommended, without a separate ruling: the canonical frames as the gauge (V1); the gauge-free readouts first, the gauge-dependent
ones stated as such (V2).

## 9. What remains

**Open decisions (the maintainers'):**
- **The filters' span.** A filter writes a whole new recording, so it walks every chunk today. Should it also default to one chunk?
- **The phase gradient's default signal:** the normal current (today's default), the ambient components (an option), or the curl of the current
  (nxr's `core_phase` reads the curl).
- **The time scale of travelling-wave work:** event-level (the band's analytic signal frame by frame, as `core_phase`), or windows of seconds
  (a series of cross-spectra), or the time average as built.
- **The transported resultant (V3):** per member, Σa·(c1 + i·c2)·e^{−iρ} about the member's centre (nxr `lc_resultant`). Exact per member; it does
  not roll up to a coarser member, so the lookup must refuse its roll-up. Recommended, not built.

**Not built:**
- **The frame's spread per member:** how far the frame turns, against parallel transport, inside each member (Rheome's `spherepatches` measure), so
  the table can say where a frame sum is meaningful. Singular frame vertices get weight 0 and their area is reported.
- **Time-resolved Fields** ([frames × vertices]): measured frame by frame with the same functions, on array-backed rows (D96) whose `within` Selection
  is the frame tiling. The Studio's time readouts (turn, speed, wave) follow on the same rows (V4).
- **Tiles across subjects:** they join only on the default subject's group tree, carried to each subject through the registered sphere. The
  subject-to-default map is not in nxr-neuro.
- **`voronoi` per piece:** the weights are taken on the whole surface; the Studio's kit takes them per piece (the mollification's last digits differ).
- **Optical flow's per-vertex time summary:** the Stokes quaternion of the velocity needs a rotation estimator (the analytic signal, or v × v̇).
- **The phase gradient's region readouts** in `measure` (Σ H / Σ P over a member), and its divergence and curl (sources, sinks, rotations) and phase
  singularities.
- **The source pipelines read the stored C:** `sourcepsd`, `sourcetensor`, `quadrature` and `hodgepower` still compute C from the recording themselves.
- **The Studio:** its schema copy of `hermitian3` (#191, open) and a layer that draws a `hermitian3` Field (ideas in).
- **nxr:** the same read-size fix as nxr-neuro #69 (its `ChunkSource(chunk_samples=...)`).

**Later: multivariate measurements within a member.** MEG fingerprinting's per-region time series are Brainstorm's scout PCA or MNE's `pca_flip`.
- **The sufficient statistic is the sensor covariance or cross-spectrum**, now stored (§6): the region's source matrix is K_r C K_rᵀ and is never formed.
- **The result is an Operator** (cortex → members: one weight vector per member). Its outputs are Fields on the members, which the scalar
  measurements then take.
- **Reference:** MNE `pca_flip`, with Brainstorm as a second golden.
