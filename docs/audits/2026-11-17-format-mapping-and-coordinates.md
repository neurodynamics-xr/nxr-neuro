# Audit, 2026-11-17: how external formats map onto the data model, and coordinate systems, spacing and transforms (read-only)

**Purpose.** nxr-neuro now reads most of the field's MEG/EEG and anatomy formats (`io/formats/`), but only some of what it reads reaches the
data model, and part of what reaches it is placed in the wrong coordinate system or in the wrong class. This audit compares, item by item, what
each format becomes today with what the model says it should become, and checks nxr-neuro's handling of coordinate systems, spacing, units and
transforms against the design already made in the Cortical Flow Studio and in nxr. It ends with the import gaps, ordered as PR-sized fixes, and
the open questions.

- Repos read (read-only): the Studio `nxr-cortical-flow` (main, 624c7f008), nxr (`nxr`, main, 7bd1843), nxr-neuro (main, 7ad6201).
- The Studio's design record is its decision register, `docs/superpowers/specs/2026-09-22-index-database-decisions.md` (cited below as
  "D<n>", the register's own numbers), and its schema, `backend/schema/model.sql`. nxr-neuro's `io/schema/model.sql` and nxr's
  `src/nxr/io/model.sql` are copies of it (diffed: identical apart from comments and nxr-neuro's schema-51 block).
- Nothing was run and no dataset content was read. No code is changed by this audit.

---

## 0. The model, as the rest of this audit uses it

| class | what it is (the schema's definition) | stored as | read-off kind in the Studio |
|---|---|---|---|
| manifold | the cell complex: a topology (how vertices connect) and a geometry (the metric), with its f-vector; positions are numbers in a coordinate system | `manifold`, `topology`, `topology_cell`, `geometry`, `geometry_dimension` | — |
| field | values on one manifold's cells | `field` | — |
| operator | a stored (or analytic) linear map from one manifold to another; also a transition map between two coordinate systems | `operator`, `operator_factor` | `placement` (a transition) · `differential` (one manifold to itself) · `spectral` · `linking` (`backend/src/entities/operatorKind.ts`) |
| selection | which part of a manifold is in play: a point, a span, spans, indices, a kernel, a set or partition; or a point on a space | `selection`, `selection_extent`, `selection_span` | — |
| measurement | one value per (selection member, field, measure) | `selection_measurement` | — |
| statistic | computed FROM a partition's members over a span of a field: a Welch PSD, a cross-spectrum, a histogram | `statistic` (nxr-neuro, `docs/specs/2026-11-13-statistics.md`) | — |
| space | an analytic manifold (line, plane, volume, sphere…) with no discretisation; a row anchors comparability | `space` | — |
| coordinate system | a chart on a space, with a unit; one reference per space, every other one defined by its transition from it | `coordinate_system` | — |

Two rules from the definitions decide most of the contested items below:

- **An operator is something a field is pushed through.** Its rows say what it maps from and to; the Studio reads its kind off those rows. A
  matrix that no pipeline ever applies to a field is not an operator because it happens to be square.
- **A statistic is what is computed from the members of a partition over a span.** Its row carries the span, the partition, the number of members,
  the degrees of freedom and the method's options as columns (`span_json`, `partition_selection_id`, `n_members`, `dof`, `options_json`).

---

## Part 1. The mapping review

What each format becomes today was read in the importers:
- `io/convert/recording.cpp` (FIF raw, EDF/BDF, BrainVision, EEGLAB, CTF, KIT: `import_recording`);
- the Brainstorm importer (`io/src/bst_*.cpp`);
- the template importers (`io/convert/templates.cpp`, `io/convert/trk.cpp`, `io/src/default_subject.cpp`);
- the pipelines that write MEG/EEG objects (`pipelines/src/meeg.cpp`).

The pure readers in `io/formats/` parse every field; what is lost is lost in the importer.

### 1.0 The contested items: verdicts

| item | verdict | reason |
|---|---|---|
| **noise / data covariance** | **a statistic** (measure `covariance`, axes channel × channel on its channels' Selection), **from which an operator, the whitener, is derived** | It is the zero-lag second moment of a field over a time partition: the blocks are the partition's members, `nfree` / the divisor is `dof`, the span and the bad-sample mask are the statistic's span. It is the zero-lag case of the sensor cross-spectrum, which nxr-neuro moved from an Operator to a statistic on 2026-11-13 (`cross_spectral_density`, `pipelines/src/meeg.cpp`), and the fingerprinting spec already stores a span covariance as a statistic (`docs/specs/2026-11-16-meg-fingerprinting.md`, §3). **No pipeline applies C to a field.** Every consumer builds something from it: the whitener C^(-1/2), the kernel's regularised inverse, a region's source covariance K C Kᵀ, an eigendecomposition. Stored as an operator, its provenance (recording, span, blocks, divisor) sits in free-form `producer_json`, and the Studio's own rule reads a channels → channels operator as `differential`, which a covariance is not. The Studio's image scope already shows it as its values on channel × channel, which is how a statistic is shown. The register's O16 made it an operator (D40) to give it a row of its own instead of an `of` sidecar. That predates the statistic table, and a statistic gives it the row. |
| **whitener** | **an operator** (channels → channels, symmetric [M × M]; or channels → its rank, PCA [r × M]), derived from the covariance statistic | It is applied to data and to the leadfield (W x, W G): a linear map in the model's sense. Its link to the statistic it comes from needs a column the schema lacks (open question Q1). |
| **SSP projectors** | **operators**, kind `projector`, channels → channels, one per projector in the file's order (as D168 imports Brainstorm's) | Applied to the samples (x → P x) and to the gain. The active ones are applied at import, as the Brainstorm importer does, and the recording records which. An orthogonal projector is idempotent (P² = P), so applying one to samples that already carry it changes nothing. That settles FIF's ambiguity over whether "active" means "already applied". |
| **CTF compensation** | **an operator**, kind `ctf compensation`, channels → channels (csr, MEG rows × reference columns) with its grade (D168) | It is applied to the gain (G − C·G_ref) and, to change grade, to the data. It is not applied at import. A recording that is not at its destination grade is refused, as the Brainstorm importer refuses it. |
| **forward solution** | **an operator**, kind `forward`, source → channels; `from_value_type` vector3 (free) or scalar (fixed); `to_selection_id` the channels it models, `from_selection_id` the source space's used vertices | It is applied to source fields (the D105 simulation). It is paired with every kernel derived from it (`inverts_id`, `inverse_kind = regularized`, D106). |
| **inverse kernel** | **an operator**, kind `inverse kernel`, channels → source, paired with its forward | As today. An MNE `-inv.fif` is not a kernel: its parts are operators (eigen-leads, eigen-fields; the composition through `operator_factor`), its singular values a field on the rank axis, its noise covariance a statistic, its source covariance and depth prior fields on the source vertices (priors, not data statistics). The kernel is made from them by a method (`meeg/`), as `docs/specs/2026-11-17-interoperability.md` §4.2 already says. |
| **source space** | **a Selection** (`indices`) of the surface it was made on (MNE `vertno`, `inuse`); a volume source space a lattice manifold with an `indices` Selection of its used voxels; a discrete one a points manifold | The model's rule is that a subset of vertices that is not a connected component is a Selection, never a manifold (D30, restated by D158). A Brainstorm source space is the whole cortex surface: the surface itself, all vertices used. A decimated mesh with its own triangles (MNE `use_tris`) is a different cell complex. It becomes a sibling manifold only when a computation needs its own topology, and the Selection says which vertices it keeps (Q7). |
| **digitisation, head-shape points, fiducials, HPI coils** | **a points manifold** `<channels>_headpoints` in the frame they were digitised in, with partitions of their kind (cardinal, HPI, EEG, extra) and label (NAS, LPA, RPA, the coil numbers) | The D166 shape. The fiducials are also what DEFINES the subject coordinate system (SCS, below): the importer derives the head → SCS transition from them, and the points stay as measured. |
| **device → head, head → MRI, CTF head → head** | **transition operators** (`kind = transition`, 3 × 4 or 4 × 4 affine, `from/to_coordinate_system_id`) between coordinate systems of the subject's head space; copied into `producer_json["transforms"]` for provenance | D25 and D40 made every affine between two charts of one space a transition operator (`T1_scs` is one). D166 kept Brainstorm's `TransfMeg`/`TransfEeg` in `producer_json` because they are a labelled chain of steps whose intermediate frames are not named. FIF's `dev_head_t`, `-trans.fif`'s head → MRI and CTF's device → head are single affines between named frames, which is exactly the transition's shape. The device frame differs per acquisition (the head moves between runs), so its coordinate system is per acquisition (Q2). |
| **channel positions and orientations** | the sensor **points manifold** (one point per channel) in the coordinate system of the frame the file gives them in, plus `<channels>_orientation` (vector3 field), `<channels>_coils` (points manifold) and `<channels>_coil_weights` (csr operator, coils → channels) | The D166 shape. The positions must be in a named, true coordinate system. Today they are labelled `scs` whatever their frame (gap G1). |
| **bad channels** | a code in the channel `quality` partition (on the sensor manifold), and the recording's flags | As today. |
| **events vs annotations** | **one spans Selection** `<rec>_events` on the recording's time line (instants: start = stop), coded by a dictionary of labels; `BAD_*` and gaps in `<rec>_bad_segments` | An annotation and an event are both labelled spans of the time axis (`docs/audits/2026-11-13-time-event-labelling.md`). Events found on a stimulus channel are a detection (family A, temporal spans), not an import: the stimulus channel is imported as a channel. An annotation naming channels is a box on sensor × time (D97). Today the channel scope is dropped (gap G12). |
| **sensor sets per type** (magnetometers, gradiometers, EEG, …) | **partitions** of the one sensor manifold (the `type` family, Brainstorm's vocabulary `MEG MAG`, `MEG GRAD`, `MEG`, `MEG REF`, `EEG`, …), plus a `coil type` family | The channels of a recording are one manifold (D73). A type is a Selection, never a manifold of its own. The coil type (`VV_MAG_T3`, `CTF_GRAD`, …) is what the forward model needs, so it is stored. |
| **reference channels** | members of the same sensor manifold, typed `MEG REF` | The compensation operator's columns name them. |

### 1.1 FIF raw (`.fif`, through `fif::RawReader` and `import_recording`)

| item | today | should be | reason | gap |
|---|---|---|---|---|
| samples (`cal × range`, SI) | recording Field on sensor × time (float32, float64 kept) | same | — | — |
| `sfreq`, `first_samp` | time line in `clock` (spacing 1/sfreq, origin first_samp/sfreq) | same | D28/D29 closed form | — |
| `chs[].loc[0:3]` | sensor manifold positions in `scs` (m) | positions in the frame the file holds them in. MEG in the **device** frame, EEG in the **head** frame. Either both carried into head by `dev_head_t`, or the manifold in head with the MEG positions moved. The coordinate system named for it; `scs` only after the fiducial-defined head → SCS transition | Neuromag head (x toward the right ear, y toward the nasion) and SCS (x toward the nasion, y toward the left ear) differ by a rotation, and the device frame by the head position | **G1** |
| `chs[].loc[9:12]` | `<channels>_orientation` | same, in the same frame as the positions | — | G1 |
| `chs[].coil_type` | reader only (`Channel::coil_type`); not stored | a `coil type` family on the sensor manifold; `_coils`, `_coil_weights` from `coil_def.dat` | the forward model integrates over coils | **G6** |
| `chs[].kind` → type | `MEG GRAD` when unit T/m, else `MEG`; `meg_ref` → `Meg_ref` | `MEG MAG` / `MEG GRAD` / `MEG REF` by coil type (Brainstorm's vocabulary) | the whitener's per-type blocks and `--types` read these names | **G6** |
| `bads` | `quality` partition | same | — | — |
| `dev_head_t`, `ctf_head_t`, `dev_ctf_t` | `producer_json["transforms"]` of the recording only | transition operators between `device:<acq>`, `head` (and `ctf-head`), plus the producer copy | D25/D40 | **G2** |
| `dig` | `producer_json["digitisation"]` only | `<channels>_headpoints` + kind/ident partitions | D166 | **G3** |
| `projs` | description, active, n_vectors in `producer_json`; matrices dropped; not applied | `<channels>_projector_<k>` operators; active ones applied at import and recorded | D168; the Brainstorm importer applies them | **G4** |
| `comps`, compensation grade | grade only | `<channels>_ctf_compensation` operator with grade; non-destination grade refused | D168 | **G5** |
| annotations | `<rec>_events` / `<rec>_bad_segments` spans (onset + t0) | same; channel-scoped annotations as boxes on sensor × time | — | G12 |
| skips | zeros + `BAD_skip` span | same | — | — |
| `meas_date`, `experimenter`, `description`, `subject_info` | kept unless `--anonymize` (meas_date, experimenter, description); `subject_info` never copied | per interoperability spec §3.2 (dates: its open question 5) | — | — |
| `line_freq`, `highpass`, `lowpass` | producer record | same (an acquisition attribute) | — | — |
| HPI, helium, Maxfilter history, cross-talk, fine calibration | `other_tags` in the reader; not imported | kept in the producer record as now; cross-talk and fine-calibration matrices become operators when a method uses them | none uses them yet | — |

### 1.2 FIF object files (`fif_objects.h`: read only; no importer)

| item | today | should be | reason | gap |
|---|---|---|---|---|
| `-src.fif` surface | not imported | the surface manifold (all `np` vertices, metres, in its frame: MRI = surface RAS) and an `indices` Selection of `vertno`; `use_tris` a sibling manifold only when needed | D30, D158 | **G8** |
| `-src.fif` volume | not imported | a lattice manifold (closed form, `vox_mri_t` its transition) + `indices` Selection of `inuse` | D26: a lattice in its voxel chart, placed by a transition | G8 |
| `-src.fif` discrete | not imported | a points manifold | — | G8 |
| `-trans.fif` (head ↔ MRI) | not imported | transition operator `head` → `freesurfer-tkr` | D25/D40 | **G2** |
| `-cov.fif` | not imported (the spec §4.2 plans an operator) | covariance **statistic** (`n_members` / `dof` from `nfree`; its bads, projectors and method in `options_json`; `of_field_id` null when the recording is not in the package) | §1.0 | **G7** |
| `-fwd.fif` | not imported | `forward` operator with its channels Selection and source-space Selection | §1.0 | G8 |
| `-inv.fif` | not imported | parts as operators and fields, noise covariance as a statistic (§1.0); no kernel at import | §1.0 | G8 |
| `-eve.fif` | not imported | instants in `<rec>_events` | — | G8 |
| `-annot.fif` | not imported | `<rec>_events` / `<rec>_bad_segments` | — | G8 |

### 1.3 EDF, EDF+, BDF, BDF+

| item | today | should be | reason | gap |
|---|---|---|---|---|
| signals, physical/digital ranges | recording Field, SI | same | — | — |
| several sampling rates | refused unless one is named | same | — | — |
| TAL annotations | `<rec>_events` | same | — | — |
| EDF+D gaps | zeros + `BAD_skip` | same | — | — |
| channel positions | none in the file; the sensor manifold has no positions (NaN) | a sensor manifold with no geometry (`geometry_id` null: no place), or positions from a BIDS `electrodes.tsv` | a manifold with no metric has no geometry row (D29); NaN positions in a stored geometry say "placed somewhere unknown" | **G1** |
| patient / recording identification | dropped under `--anonymize` (by key name) | never copied (spec §3.2) | — | — |

### 1.4 BrainVision

| item | today | should be | reason | gap |
|---|---|---|---|---|
| samples, resolution, units | recording Field, SI | same | — | — |
| markers, `New Segment` | `<rec>_events`, gaps | same | — | — |
| `[Coordinates]` (radius, θ, φ) | Cartesian on the sphere of the given radius; `position_unit` empty, **treated as metres**; labelled `scs` | a coordinate system `brainvision-sphere` on head space with no transition (not comparable to the anatomy until fitted), the radius' unit recorded as unknown; or no geometry | a unit sphere is not metres, and the electrodes are not in SCS | **G1** |

### 1.5 EEGLAB

| item | today | should be | reason | gap |
|---|---|---|---|---|
| data (`.fdt` or inside `.set`) | recording Field | same | — | — |
| `EEG.event`, `boundary` | `<rec>_events`, gaps | same | — | — |
| `chanlocs` X/Y/Z | `position_unit` empty, **treated as metres**; labelled `scs` | a coordinate system `eeglab` (x toward the nose, y left, z up; unit as `EEG.chaninfo` states it, else unknown), with no transition unless fiducials are present | EEGLAB's axes resemble SCS but its origin and unit are the file's | **G1** |
| `icaweights`, `icasphere`, `icawinv` | not imported | operators recorded, not applied (spec §4.5); the ICA projector form of D168 when components are marked removed | — | G4 |

### 1.6 CTF `.ds`

| item | today | should be | reason | gap |
|---|---|---|---|---|
| `.meg4` samples | recording Field | same | — | — |
| channel coils (first coil's position and normal, device frame) | sensor manifold labelled `scs`; orientations | positions in `ctf-head` (= Brainstorm's SCS convention, defined by the `.hc` coils) by the reader's device → head transform, in `scs` only then; `_coils` and `_coil_weights` from all coils | CTF head and SCS share their axis convention; the device frame does not | **G1**, G6 |
| `.hc` head coils, `transforms()` | dropped | `_headpoints` (coils, kind HPI/cardinal) and transition operators `device:<acq>` → `ctf-head` | D166, D25 | **G2**, **G3** |
| compensation records, grade | grade only | `<channels>_ctf_compensation` operator; grade checked | D168 | **G5** |
| `MarkerFile.mrk`, `bad.segments`, `BadChannels` | events, bad segments, quality | same | — | — |
| `.pos` / `.eeg` electrode files | read, not imported | EEG positions and head points | — | G3 |
| reference channels | kind `meg_ref` → type `Meg_ref` | `MEG REF` | Brainstorm vocabulary; the compensation names them | **G6** |

### 1.7 KIT / Yokogawa

| item | today | should be | reason | gap |
|---|---|---|---|---|
| samples | recording Field | same | — | — |
| channel positions (device) | labelled `scs` | `device:<acq>`, then head through the coregistration | — | **G1** |
| digitisation (`elp`, `hsp`, named points), marker coils, coregistration | dropped (only system and model names recorded) | `_headpoints`; the device → head transition from the marker coils fitted to the digitised HPI points (MNE's rule) | D166, D25 | **G2**, **G3** |

### 1.8 NIfTI-1/2 (read only; no importer)

| item | today | should be | reason | gap |
|---|---|---|---|---|
| volume | not imported (readers and bindings only) | volume manifold, closed form in `voxel:<name>`, a Field of intensities | D26, D77 | **G9** |
| sform/qform | reader gives the best affine | a transition operator `voxel:<name>` → the volume's world system. That is scanner RAS (or the template's, MNI for a template file) in mm, with `qform_code`/`sform_code` recorded. Not `scs` until a scanner RAS → SCS transition exists (from fiducials) | the world frame of a NIfTI file is its own, stated by its codes | G9 |
| label volume | — | a codes Selection, only when its parcellation is named | importer rule | G9 |

### 1.9 GIFTI (read only; no importer)

| item | today | should be | reason | gap |
|---|---|---|---|---|
| `.surf.gii` | not imported | surface manifold in the coordinate system the file's `CoordinateSystemTransformMatrix` names (`NIFTI_XFORM_*`), metres | — | **G10** |
| `.func.gii` / `.shape.gii` | not imported | Fields on a named surface (vertex count checked) | — | G10 |
| `.label.gii` | not imported | parcellation codes Selection (named only) | importer rule | G10 |

### 1.10 CIFTI-2 (read only; no importer)

| item | today | should be | reason | gap |
|---|---|---|---|---|
| brain models (surface vertices) | not imported | Fields on the matching template surfaces, through an `indices` Selection of the vertices the file covers (the medial wall is left out) | the file covers a subset of a template's vertices: a Selection | **G11** |
| brain models (voxels) | not imported | a Field on the template volume through an `indices` Selection of voxels | same | G11 |
| `dtseries` | not imported | Field on (selection of surface) × time, streamed by rows | — | G11 |
| `dlabel` / parcels | not imported | codes Selection; parcel series as measurements on its members | D158 | G11 |

### 1.11 FreeSurfer surfaces and annotations

| item | today | should be | reason | gap |
|---|---|---|---|---|
| `lh/rh.<surf>` (fsaverage template) | surface in coordinate system `fsaverage`. That key falls through to a space of its own, **with no unit**; positions are tkr mm/1000 = metres | `freesurfer-tkr` (unit m) on the template's head space. For fsaverage, also the documented MNI305 relation as a transition | the generic arm of the coordinate-system seed (`io/src/subject_writer.cpp`, as nxr's `_space_spec`) gives an unrelated space; nxr's MNE converters use `freesurfer-tkr` for the same surfaces | **G13** |
| `sphere.reg` | sibling geometry in `freesurfer-sphere` | same | D26 | — |
| `.annot`, `.label` | parcellation codes Selections, named only | same | D163 | — |
| a subject's own FreeSurfer folder | not imported (spec: in progress) | surfaces in `freesurfer-tkr`, with the `c_ras` offset (from `orig.mgz`) as the transition to scanner RAS | so the surfaces meet a NIfTI T1 and the `-trans.fif` MRI frame | G10 |

### 1.12 DSI Studio / TrackVis `.trk`

| item | today | should be | reason | gap |
|---|---|---|---|---|
| streamlines | curve manifold in `scs` (m), after voxel-mm → RAS mm (as nibabel), a half-voxel shift to MNI mm and MNI → SCS by the anatomy's NCS and SCS frames, all baked in | same positions; the MNI → SCS map stored once as a transition (`mni` → `scs`) instead of only described in the producer record | D25: the map is reusable (electrodes in MNI, NIfTI templates) | G13 |
| ends on a surface | csr ends operator (D125) | same | — | — |

### 1.13 BIDS (read only; no importer)

| item | today | should be | reason | gap |
|---|---|---|---|---|
| entities | parsed by the reader | the `acquisition` row (D161) | — | **G14** |
| recordings | — | through the recording importer | — | G14 |
| `channels.tsv` status, type, units | — | quality and type partitions | — | G14 |
| `events.tsv` | — | `<rec>_events` | — | G14 |
| `*_coordsystem.json`, `electrodes.tsv`, `headshape` | — | coordinate systems named for the BIDS coordinate system (`CTF`, `ElektaNeuromag`, `CapTrak`, a template such as `MNI152NLin2009cAsym`) with their unit, transitions from the stated anatomical landmarks; positions and head points in them | nxr's BIDS-iEEG converter already names the coordinate system after the BIDS space | G14 |
| `dataset_description.json` | — | the origin record | spec §3.2 | G14 |

### 1.14 Brainstorm importer (`io/src/bst_*.cpp`)

| item | today | should be | reason | gap |
|---|---|---|---|---|
| channel file `Loc`, `Orient`, coils, weights, head points | sensor manifold in `scs` (Brainstorm stores them in SCS), D166 shape | same | — | — |
| `TransfMeg` / `TransfEeg`, `SCS` fiducials | sensor manifold `producer_json["transforms"]` | as D166; the fiducials also give the subject's SCS definition | the chain's intermediate frames are not named (D166) | — |
| `Projector`, `MegRefCoef` | operators, active applied (D168, `bst_projectors.cpp`) | same | — | — |
| head model `Gain`, `GridLoc`, `GridOrient` | `forward` operator; `_GridLoc`/`_GridOrient` fields of it | same; a volume head model's grid as a lattice or points manifold, not fields on the cortex | — | — |
| kernel `ImagingKernel` | `inverse kernel` operator paired with its forward | same | — | — |
| `NoiseCovMat.NoiseCov` | `<kernel>_noise_cov` operator, kind `noise covariance` | covariance **statistic** (with `nAvg` as `n_members` where present) | §1.0 | **G7** |
| kernel `Whitener` | `<kernel>_whitener` operator, `derived_from` the covariance operator | operator, derived from the statistic | §1.0 | G7 |
| MRI `Cube`, `SCS`, `NCS` | volume in `voxel:<vol>`, intensities Field, `T1_scs` / `T1_ncs` transition operators (voxel → mm) | same, with the unit question settled (Q5) | — | Q5 |
| surfaces | `scs`, metres | same | — | — |
| `timefreq_*` (PSD, TF) | Fields | statistics: Brainstorm's `timefreq` file is the statistic's model (`docs/specs/2026-11-13-statistics.md`) | the statistic spec | **G15** |
| events, bad segments, atlases, fibres, maps | spans Selections, codes Selections, curve + ends operator, Fields | same | — | — |

### 1.15 nxr-neuro's own computations (for completeness)

| item | today | should be | gap |
|---|---|---|---|
| `noisecov compute` | Operator `noise covariance` `<R>_noise_cov`, span and blocks in `producer_json` | statistic (§1.0) | **G7** |
| `inverse compute` whitener | operator | operator derived from the statistic | G7 |
| `csd compute` | statistic (2026-11-13); the old Operator form still read | same | — |

---

## Part 2. Coordinate systems, spacing, units, transforms

### 2.1 The design as found (the Studio and nxr)

**Spaces, coordinate systems and manifolds** (D24–D29, `backend/schema/model.sql`, `backend/src/entities/space.ts`).

- A **space** is an analytic manifold: a parametric geometry (line, circle, plane, sphere, torus, volume) with its parameters and no
  discretisation. It stores no coordinates. Two rows are two copies of one geometry. A row anchors **comparability**: two coordinate systems
  of one space are related by a transition map, two spaces are not. Spaces are per subject today. Shared spaces (a template's, a spectral
  line) are left open with the templates question.
- A **coordinate system** is a chart on a space, with a unit. One per space is the reference (`is_reference`, transition `identity`). Every
  other one is defined by its transition FROM the reference: `identity`, `affine`, `log`, `power` or `operator`. A non-isometric system
  (log, power) presents the space in other numbers and changes no manifold's metric.
- A **manifold's positions are always numbers in one coordinate system** (`geometry.coordinate_system_id`). Its unit is the system's, and
  its extrinsic dimension is the space's rank. A rule (an affine, a log) is a coordinate system. Coordinates stored per vertex (a flat map,
  a sphere) are a sibling geometry sharing the topology (D26).
- **The seed** (`spacesFor` in the Studio's `backend/src/db/database.ts`; `_space_spec` in nxr's `src/nxr/io/writer.py`; `space_spec` in
  nxr-neuro's `io/src/subject_writer.cpp`, a port):

| space | systems (reference first) | unit |
|---|---|---|
| `head` (volume, R³) | `scs` (reference: Brainstorm's subject coordinate system, origin between the ears, x toward the nasion) · `ncs` (AC · PC · IH) · `mni` · `voxel:<volume>` · `freesurfer-tkr` (nxr and nxr-neuro only) | m · mm · mm · voxel · m |
| `sphere` (volume) | `freesurfer` (radius 100) | — |
| `time` (line) | `clock` | s |
| `frequency` (line) | `hz` · `octave` (log, base 2) | Hz · log2 Hz |
| `spectrum:<operator>` (line) | `eigenvalue` · `wavelength` (power: 2π/√λ, or 2π/|λ| for a Dirac) | 1/m² or 1/m · m |
| any other key | a space of that name with one reference system of that name | none |

**Transforms.**
- **A transition map is an operator** (D25; built by D40): `kind = transition`, `from/to_coordinate_system_id` both set, a 3 × 4 affine.
  The Studio reads any operator with coordinate systems as a `placement` (`operatorKind.ts`). The built cases are the T1's `T1_scs` and
  `T1_ncs` (voxel index → SCS / NCS millimetres, `scs_mm = R · (voxel ⊙ voxsize) + T`), written by the Studio's converter, nxr
  (`src/nxr/convert/mri.py`) and nxr-neuro (`io/src/bst_mri.cpp`) alike.
- `coordinate_system.transition_json` holds a transition's parameters only until the operator exists. The register's owed item O17 says it
  is to be derived from the transition operators (ncs ← scs is `T1_ncs ∘ T1_scs⁻¹`) and the column then goes. Not built.
- **Device → head chains are not modelled** (D166). Brainstorm's `TransfMeg`/`TransfEeg` and the SCS fiducials are kept in the sensor
  manifold's `producer_json["transforms"]`, on the grounds that a labelled chain of device → head steps is neither a coordinate system nor a
  transition. No table was invented.
- **Time → frequency is not a chart change** (D25): it moves no point. It maps a field on the time line to a field on the frequency line, so
  it is an operator on values (the DFT, analytic, D93), and the frequency lattice is the time lattice's dual.

**Spacing and origin: closed form against explicit** (D28, D29, D37, D77).
- A **closed-form** geometry is one `geometry_dimension` row per array dimension: `n_vertices`, `spacing` (the smallest discretisation;
  1/spacing along time is the sampling rate), `origin` (the first vertex's coordinate, null when the producer wrote none) and `unit`, in
  the geometry's coordinate system, plus `direction_json` when the axes are not the identity. Positions are generated,
  `origin + direction · (index ⊙ spacing)`, then the space's parametrisation. The viewer of a lattice reads nothing (D77).
- A **stored** geometry has a `positions_path` (one position per vertex) and an induced metric: surfaces, sensors, curves.
- A **product** (sensor × time) has a dimension row per factor that names the factor manifold (D37).
- Time lines: closed form in `clock`, spacing 1/sfreq s, origin the first sample's time. Frequency lines: closed form in `hz`. Voxel grids:
  closed form in their own `voxel:<volume>` system, with the transition to `scs` as the operator.

**A voxel grid's affine.** The lattice is placed in its own index chart (D26). The affine is the transition operator from that chart to
`scs` (and `ncs`), 3 × 4 over the voxel INDEX with the voxel size folded in, output in millimetres. The dimension rows carry the voxel size
as spacing with unit `mm` (`bst_mri.cpp`, `mri.py`).

**Units.** Every field carries its SI unit (D39). Positions are in metres in `scs` (surfaces, sensors, fibres; the Brainstorm convention).
`ncs` and `mni` are in mm. The T1 transitions output mm. Eigenvalues are in 1/m² (Laplace–Beltrami) or 1/m (Dirac).

**MNE frames in nxr** (`src/nxr/convert/mne_meg.py`, `mne_eeg.py`).
- nxr's MNE converters work in MNE's MRI frame = FreeSurfer's surface RAS. Channels are carried there from head by the `-trans.fif`
  transform and written in `freesurfer-tkr` (m). The cortex is the subject's own white surface in the same system.
- The head → MRI transform rides in the sensor manifold's `producer_json` (`head_to_mri`), not as a transition.
- `freesurfer-tkr` is created on head space with transition `affine` and NO parameters, so it cannot be converted to `scs`.

**BIDS in nxr** (`src/nxr/convert/bids_ieeg.py`). Electrode positions are converted to metres from `iEEGCoordinateUnits`, and the coordinate
system is named for the BIDS `space` entity (a template space). The seed's generic arm makes it a space of its own.

**What the design does not cover.**
- RAS vs LPS: no system records an axis convention beyond its description.
- Scanner RAS vs surface RAS: only `freesurfer-tkr` exists; there is no `scanner-ras`, and FreeSurfer's `c_ras` is not stored.
- MNE's head and device frames: no coordinate systems.
- The per-acquisition nature of the device frame: not addressed.

### 2.2 nxr-neuro checked against it

| design point | nxr-neuro | verdict |
|---|---|---|
| spaces and systems seeded from keys (`spacesFor`) | ported exactly (`io/src/subject_writer.cpp`) | implemented |
| time line closed form in `clock`, spacing 1/sfreq, origin t0 (FIF: first_samp/sfreq) | `write_time_line` (`io/src/bst_recording.cpp`) | implemented |
| frequency lines in `hz`, eigen lines in `eigenvalue:<op>` with the wavelength power system | `bst_frequency.cpp`, `spatial_bands.cpp` | implemented |
| T1 in `voxel:<vol>` + `T1_scs`/`T1_ncs` transition operators | `bst_mri.cpp` (as nxr) | implemented, with the unit inconsistency the design has (Q5) |
| positions are numbers in the coordinate system they are in | **departs**: `import_recording` writes every format's channel positions under `scs` (`SensorOptions::cs_key` default). The positions are in device, Neuromag head, CTF device, a unit sphere or EEGLAB's frame, and an empty position unit is taken as metres (`metres_per("") = 1`) | **G1** |
| affine transitions have parameters | **departs with nxr**: `ncs`, `mni`, `freesurfer-tkr` systems are inserted with `transition = affine` and `transition_json` NULL; only the T1's transitions exist as operators, and nothing links them (O17) | G2, Q4 |
| transforms between frames as transition operators | FIF `dev_head_t`, `ctf_head_t`, `dev_ctf_t` only in the recording's `producer_json`; CTF `transforms()` and KIT coregistration dropped | **G2** |
| digitisation as `_headpoints` (D166) | FIF `dig` only in the producer record; CTF `.hc` and KIT digitisation dropped | **G3** |
| projectors and CTF compensation as operators (D168) | Brainstorm importer yes; recording importer no (description and grade only) | **G4**, **G5** |
| a template's surfaces in a known system with a unit | fsaverage in key `fsaverage`: a space of its own with no unit (`io/convert/templates.cpp`, `io/src/default_subject.cpp`) | **G13** |
| `.trk` MNI → SCS | applied and described in the producer; no `mni` → `scs` transition stored | G13 |
| covariance and cross-spectrum as statistics | cross-spectrum yes (2026-11-13); noise covariance still an operator (computed and imported) | **G7** |

---

## The import gaps, ordered as PR-sized fixes

Correctness first (data placed in the wrong frame are silently wrong); then what the model already has a shape for (D166, D168); then the
class changes; then new importers.

1. **G1: Channel positions in their true frame and unit.**
   - Carry each channel's frame out of the readers (FIF `coord_frame` is already in `fif::ChannelInfo`; CTF and KIT are device; BrainVision
     and EEGLAB are their own) into `formats::Channel` or the header.
   - Write the sensor manifold in a coordinate system named for that frame: `device:<acquisition>`, `head` (Neuromag), `ctf-head`,
     `brainvision-sphere`, `eeglab`. Write it in `scs` only after a fiducial-defined transition, as Brainstorm places sensors.
   - FIF: move MEG positions into head by `dev_head_t`, so one manifold has one frame.
   - Refuse an empty position unit as metres. Write no geometry (or record the unit as unknown) when the file states none.
   - The Brainstorm path is unchanged.
2. **G2: Transforms as transition operators.**
   - FIF `dev_head_t`, `ctf_head_t`, `dev_ctf_t`; CTF `transforms()`; KIT's marker-coil coregistration; `-trans.fif` (head → `freesurfer-tkr`).
   - Each one becomes `kind = transition`, with both coordinate systems created with their transition. Also kept in `producer_json` for
     provenance.
   - Head → SCS from the cardinal points (NAS, LPA, RPA), so sensors and anatomy meet in `scs`.
   - Depends on Q2 and Q4.
3. **G3: Digitisation as `<channels>_headpoints`** (D166 shape: points manifold + kind and label partitions).
   - FIF `dig`, CTF `.hc` coils and `.pos`, KIT `elp`/`hsp`/named points.
   - The points stay in the frame they were digitised in.
4. **G4: Projectors in the recording importer.**
   - Port `bst_projectors.cpp`'s operators to FIF `projs`: P = I − UUᵀ per projector, on the channels it names, in the file's order.
   - Apply the active ones while copying, and record which on the recording (D168's record).
   - EEGLAB ICA matrices recorded as operators, not applied.
5. **G5: CTF compensation in the recording importer.**
   - FIF `comps` and CTF `.res4` compensation records become `<channels>_ctf_compensation` (csr, MEG × REF) with `grade`.
   - Not applied. A recording that is not at its destination grade is refused, as the Brainstorm importer refuses it.
6. **G6: Channel vocabulary and coils.**
   - `MEG MAG` / `MEG GRAD` / `MEG` / `MEG REF` from the coil type (today: magnetometers `MEG`, references `Meg_ref`).
   - A `coil type` family on the sensor manifold.
   - `_coils` and `_coil_weights` from `coil_def.dat` (interoperability spec, question 7).
7. **G7: Covariance as a statistic.**
   - Register the measure `covariance` (and `sample` / `empirical` methods) in `io/schema/statistics.json`.
   - `noisecov compute` writes a statistic (span, blocks as members, divisor as `dof`, options).
   - `inverse compute` and the Brainstorm importer's `NoiseCovMat` read and write the statistic. The whitener stays an operator derived from it.
   - Readers accept the old Operator form, as was done for the cross-spectrum.
   - Amend `docs/specs/2026-11-10-neuro-forward-inverse.md` §7 and the interoperability spec §4.2.
   - Needs Q1 for the link.
8. **G8: FIF object importer.**
   - `-src` (surface manifold + `vertno` Selection; volume lattice + `inuse` Selection; discrete points).
   - `-fwd` (forward operator with both Selections).
   - `-inv` parts (operators, fields, the covariance statistic; no kernel).
   - `-cov` (statistic, through G7), `-eve` and `-annot` (events).
9. **G9: NIfTI importer.** Volume in `voxel:<name>`, intensities Field. The sform/qform as a transition to the file's world system
   (scanner RAS or a template's, by its codes). Label volumes only when named.
10. **G10: FreeSurfer subject and GIFTI importers.**
    - Surfaces in `freesurfer-tkr` (m), with `c_ras` as the transition to scanner RAS.
    - GIFTI surfaces in the system their transform matrix names.
    - Maps and labels on named surfaces.
11. **G11: CIFTI importer.** Fields through `indices` Selections of the template surfaces and volume. `dtseries` as a series on the
    selection × time. Labels and parcels as Selections and measurements.
12. **G12: Channel-scoped annotations.** As boxes on sensor × time (D97) instead of dropping the channel list.
13. **G13: Template coordinate systems.**
    - fsaverage surfaces in `freesurfer-tkr` (unit m), not the unit-less `fsaverage` key.
    - Store the `mni` → `scs` map the `.trk` importer computes as a transition operator.
    - Fill or derive `transition_json` for every affine system (O17).
14. **G14: BIDS importer.** One package per `sub-`; acquisitions from entities; channels, events, coordinate systems, electrodes and head
    shape through G1–G3; the origin from `dataset_description.json`.
15. **G15: Brainstorm `timefreq` files as statistics**, not fields (the statistic spec's own model).

## Open questions, each with a recommendation

- **Q1. How does an operator name the statistic it is derived from (the whitener from the covariance)?**
  - `operator.derived_from_id` references operators only.
  - **Recommendation:** add `operator.of_statistic_id` (and to `operator_attributes`) in the shared schema, as `selection_measurement`
    gained `of_statistic_id`, mirrored in the Studio and nxr.
  - Until then, `producer_json["covariance"]` names it.
- **Q2. Where does the per-acquisition device frame live?**
  - **Recommendation:** one coordinate system per acquisition, `device:<acquisition>`, on the subject's head space, with a transition
    operator to `head` per acquisition. Keep one head space per subject: comparability across runs is the point of a space.
  - The sensor manifold of a run is then in `head` or `scs` (one frame, comparable), its device positions recoverable by the inverse transition.
- **Q3. What is head space's reference when there is no MRI?**
  - **Recommendation:** `scs` stays the reference everywhere (the Studio's and Brainstorm's convention). Without an MRI it is defined from
    the digitised NAS, LPA and RPA, as Brainstorm does.
  - Without fiducials, the file's frame is a non-reference system with no transition. Its positions are explicitly not comparable to
    anything until one is given.
- **Q4. One home for transition parameters.**
  - **Recommendation:** O17 as written. Transition operators are the one home. `transition_json` is derived from them (or dropped), and a
    system is created only with its transition, never `affine` with no parameters.
- **Q5. Units of the voxel chart and its transition.**
  - The T1's dimension rows give spacing in `mm` in a system whose unit is `voxel`. The transition outputs SCS millimetres, while `scs`'s
    unit is `m`.
  - **Recommendation:** spacing 1 with unit `voxel` (the voxel size lives in the affine), and transitions that output the target system's
    unit (m for `scs`).
  - This changes all three repos and needs a migration of stored T1 rows. Decide together, or record "transitions output mm" as the rule.
- **Q6. Imported covariances with no recording in the package** (`-cov.fif`, Brainstorm `NoiseCovMat`).
  - **Recommendation:** a statistic with `of_field_id` null and its provenance in `options_json` (`nfree` or `nAvg` as `n_members`, method,
    bads, projectors).
- **Q7. MNE's decimated source surface (`use_tris`): Selection or manifold?**
  - **Recommendation:** an `indices` Selection of the full surface by default (D30, D158). Make a sibling manifold only when a computation
    needs the decimated mesh's own topology, and link it to the full surface through the Selection.
  - nxr's MNE converters store the decimated mesh as the cortex. That is a choice to align on.
- **Q8. RAS, LPS, scanner RAS.**
  - **Recommendation:** store RAS-family systems only. Convert LPS (DICOM, ITK) at the reader boundary and record it.
  - Add `scanner-ras` (m) on head space, with `freesurfer-tkr` related to it by `c_ras`.
- **Q9. Positions in mm systems.**
  - **Recommendation:** keep `ncs` and `mni` in mm, as the systems' units say (the transition carries the scale).
  - Every position a writer stores is in metres unless its system's unit says otherwise, and a reader converts through the transition,
    never by assuming.

## For the coming type system

- **A manifold port's type includes its coordinate system** (and through it its space and unit), not only its kind, rank and vertex count.
  - Two sensor manifolds in different frames (device vs head) are different types.
  - A node that needs positions in `scs` (a forward model, a sensor-to-cortex projection) accepts the other only through a transition, which
    the executor composes when one exists and refuses with the reason when none does (positions with no transition are not comparable).
  - Today every sensor manifold claims `scs`, so such a check would pass on wrong data: G1 comes before the type check.
- **A field port's type:** (manifold type, cell key, value type, number of components, unit). Units are checkable once G1 removes the
  "empty unit means metres" path.
- **An operator port's type:** (from manifold, from cell, from value type, from selection) → (to …).
  - A transition operator's type is (from coordinate system) → (to coordinate system) on one space. It is a separate port kind, since the
    Studio already reads it as a placement, not a mapping.
  - The pair and chain checks (`backend/src/db/pairs.ts`, D106) are the type rules for compositions.
- **A selection port's type:** (manifold, cell key, selection type). A source space as a Selection makes "kernel rows" a typed restriction
  of the cortex instead of a separate manifold.
- **A statistic port's type:** (measure, method, axes: one manifold or selection per axis, value type).
  - With the covariance a statistic, the whitener node has a statistic input (`covariance`, channels × channels on Selection S) and an
    operator output.
  - The kernel builder takes a forward operator and a covariance statistic on the same channel Selection, a check the types can make. Today
    it is a name convention (`<kernel>_noise_cov`).
- **Time and frequency lines** are typed by their space (`time`, `frequency`) and system (`clock`, `hz`). Two recordings' lines are
  comparable because they share the `time` space. A Welch statistic's frequency axis is `derived_from` the Frequency manifold (D158).
