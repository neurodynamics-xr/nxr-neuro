# Interoperability: native readers and writers for the field's formats, the exporters, and the Python and MATLAB faces (spec, 2026-11-17)

Status: **design, no code.** It turns D6 of `2026-11-16-distribution-and-interoperability.md` (exports first: GIFTI, TSV, MNE objects; then
BIDS-derivatives; imports from BIDS MEG/EEG and FIF) and the first slice of D3 (the Python face) into a plan. It builds on
`2026-11-16-subject-centred-datasets.md` (what a subject package holds; importers write packages; every item carries an `origin` record; the
importer rule, section 4.5) and `2026-11-16-command-line.md` (the verb-first grammar). Open questions are in section 8, each with a recommendation.

## Amendment (2026-11-17): the faces' verbs are `compute` and `run`

The bindings' levels are redrawn by `2026-11-17-compute-and-run.md` (decided by the maintainers the same day). Section 6 and question 13 below are read
with these changes:
- the pure-function level `nn.ops` (Python) and `nxr.ops` (MATLAB) becomes **`nn.compute` / `nxr.compute`**: `nn.compute.<group>.<node>(arrays, …)` for one
  node and `nn.compute(graph, arrays, …)` for a graph or a composite node, generated from the node registry, so what can be computed alone is what can be
  chained in a definition;
- the planned pipeline level `nn.compute.<verb>(sub, …)` and `nxrneuro.compute('psd', sub, …)` becomes **`nn.run('<pipeline>', dataset, subject, …)`** and
  **`nxr.run(...)`**, the command line's `run`;
- the MATLAB package is `nxr` (as the first slice built it), not `nxrneuro`.

## 1. Why, and the rules this follows

nxr-neuro measures the spatiotemporal patterns of cortical activity. Its results are worth something to the field only if the data can come in
from the files other tools write, and the measurements can go out to the tools the field already uses (Workbench, freeview, MNE-Python, a
spreadsheet, a BIDS-derivatives archive). Today only Brainstorm protocols and meshes come in, and only measurement tables go out.

Rules in force (decided by the maintainers, 2026-11-16):
- **Native.** nxr-neuro imports and converts popular formats itself, in C++: no MATLAB and no Python at run time. Python and MATLAB programs
  are references for parity (they write the synthetic test files and read them back for the goldens), never run-time dependencies.
- **Every importer writes an `origin` record** (`io/include/nxr/io/subject_package.h`, `OriginRecord`: source, licence, attribution,
  whether it may be redistributed) on every item it writes.
- **The importer rules** (subject-centred spec, 4.5): parcellations are imported only when the caller names them; a subject without its own
  MRI keeps the default anatomy its data were mapped on, as its own; a dataset's template is the template package its data were computed in,
  never a silent default.
- **The faces.** A Python wheel (nanobind + scikit-build-core) and a MEX binding over the **same library API** replace the project's Python
  and MATLAB toolboxes later (D1, D3).
- **In progress elsewhere, not duplicated here:** FreeSurfer surfaces and annotations, and DSI Studio TrackVis `.trk` streamlines, are being
  written now in `io/convert/` (namespace `nxr::convert`, with the template importers and their source manifest). This spec lists them as
  in progress and builds on the same pattern.

And the rules every port follows (`ARCHITECTURE.md`, `CONTRIBUTING.md`): the layers and the dependency rule; recordings streamed one 2^15-sample
chunk at a time through io's chunk service, the span always the caller's; the method rule (each reader proven against a named reference by
goldens on **synthetic** files, never a real recording); no GPL code (and no LGPL linked statically); descriptive names; stored precision
follows the input.

## 2. The survey: libraries or native

Checked 2026-11-16. Licences were read from each project's own LICENSE (or COPYING) file; "last push" is the repository's, from GitHub or
GitLab. The licence of nxr-neuro is BSD-3-Clause; compatible licences are MIT, BSD, Apache-2.0, MPL-2.0, zlib and public domain.

### 2.1 Candidates

| format | candidate | licence (from its LICENSE) | maintenance | size, dependencies | streams a long recording | Linux / macOS / WebAssembly | writes |
|---|---|---|---|---|---|---|---|
| FIF (raw, cov, fwd, inv, source spaces, evoked) | MNE-CPP `src/libraries/fiff` (`mne-tools/mne-cpp`, v2.4.0) | BSD-3-Clause | active (pushed 2026-11) | ~70 files; **requires Qt (Core, Network)**: `find_package(Qt6/Qt5 REQUIRED)` in its CMake; `FiffStream` derives from `QDataStream` over a `QIODevice` | yes (tag directory, data buffers) | Linux/macOS with Qt; WebAssembly only with a Qt-for-wasm build | yes |
| EDF/EDF+/BDF/BDF+ | EDFlib (`gitlab.com/Teuniz/EDFlib`) | BSD-3-Clause | one maintainer; last commit 2025-02 | one C file (~6,900 lines) + header; no dependencies | yes (`edfseek`, `edfread_physical_samples` per signal) | builds anywhere | yes |
| BrainVision (`.vhdr`, `.vmrk`, `.eeg`) | no maintained C/C++ library found | — | — | — | — | — | — |
| EEGLAB `.set` / `.fdt` | matio (already a dependency, v1.5.28) | BSD-2-Clause | active (pushed 2026-11) | in the tree; HDF5 for v7.3 files | `.fdt` by seek; an embedded `EEG.data` through matio's slab reads | in the tree (WebAssembly without v7.3) | — (not needed) |
| NIfTI-1/2 | nifti_clib (`NIFTI-Imaging/nifti_clib`, v3.0.0 of 2020) | public domain | low (last push 2024-12) | C, file-based I/O through its `znzlib`, global options | volumes are small | builds anywhere | yes |
| GIFTI | gifticlib (`NIFTI-Imaging/gifti_clib`) | public domain | dormant (last push 2021-02) | C; needs **expat**, zlib and nifti_clib | not needed (maps are small) | builds anywhere | yes |
| CIFTI-2 | CiftiLib (`Washington-University/CiftiLib`) | BSD-2-Clause | active (pushed 2026-09) | C++; needs **Qt (QtCore, QtXml)**, or **Boost (filesystem) and libxml++** (LGPL) | yes (row reads) | heavy; not WebAssembly without porting | yes |
| XML (for GIFTI, CIFTI) | pugixml (`zeux/pugixml`, v1.16) | MIT | active (pushed 2026-10) | two source files, no dependencies | DOM (fine for maps) | builds anywhere, WebAssembly included | yes |
| TRX tractography | trx-cpp (`tee-ar-ex/trx-cpp`, 0.6) | BSD-2-Clause | young, active (pushed 2026-09) | needs **libzip** (fetched), Eigen, and bundles json11 and mio (memory mapping) | memory-mapped | Linux/macOS; memory mapping does not carry to WebAssembly | yes |
| XDF | libxdf (`xdf-modules/libxdf`, v1.0.4) | BSD-2-Clause | maintained (pushed 2026-07) | C++, bundles pugixml | **no**: `load_xdf` reads the whole file into `std::vector<std::vector<float>>` | builds anywhere | no |
| CTF, KIT/Yokogawa, 4D/BTi MEG | no maintained permissive C/C++ library found; FieldTrip's readers are GPL-3.0 (never copied) | — | — | — | — | — | — |
| BIDS (layout, `participants.tsv`, sidecars) | no C/C++ library needed: paths, TSV and JSON | — | — | nlohmann/json (in the tree) | — | — | — |
| MNE `-stc` source estimates | (part of FIF tooling; the `.stc` file is a small big-endian binary) | — | — | — | — | — | — |

Python bindings (section 6): **nanobind** (`wjakob/nanobind`, BSD-3-Clause, latest tag v3.1.0, active), **scikit-build-core**
(`scikit-build/scikit-build-core`, Apache-2.0, v1.1.2 of 2026-11-02), **cibuildwheel** (`pypa/cibuildwheel`, BSD-2-Clause, v4.3.5 of
2026-10-27; a build tool, not distributed).

### 2.2 Verdicts

| format | verdict | why |
|---|---|---|
| **FIF** (raw, cov, fwd, inv, `-src`, `-trans`, `-eve`, evoked) | **native**, porting the logic of MNE-Python's FIF reader and writer (BSD-3-Clause; ported with attribution in `THIRD_PARTY_NOTICES.md`) | MNE-CPP is BSD and good, but needs Qt, which is LGPL-3/GPL/commercial: it cannot be linked statically into the one binary or the wheel under our rules, is very large, and has no simple WebAssembly build. The FIF tag layer (tag = kind, type, size, next; big-endian; a directory of tags; nested blocks) is small; the work is in the objects, and MNE-Python is the reference for them anyway |
| **EDF/BDF** (+ EDF+/BDF+) | **native** | EDFlib is BSD-3 and battle-tested, but it refuses discontinuous EDF+D/BDF+D files (`EDFLIB_FILE_IS_DISCONTINUOUS`), keeps a global table of open files and is not thread-safe ("use a mutex"), and reads only from a path (no memory buffer for unit tests or the browser). The format is a fixed ASCII header and records of 16- or 24-bit integers, and the EDF+ annotation lists (TAL) are a short grammar: a native reader is about as short as wrapping EDFlib, and keeps gaps. EDFlib stays a second reference in its parity tests |
| **BrainVision** | **native** | an INI-like text header and marker file, and one binary of int16, int32 or float32, multiplexed or vectorized. Nothing to reuse |
| **EEGLAB** | **native over matio** (in the tree) | the `.set` is a MATLAB file (v5 through matio; v7.3 through matio + HDF5); the `.fdt` is float32 by seek |
| **NIfTI-1/2** | **native I/O over nifti_clib's public-domain header definitions** (`nifti1.h`, `nifti2.h` vendored) | the I/O is a 348- or 540-byte header, optional extensions and an array, gzipped or not (zlib is in the tree). nifti_clib's I/O is file-based with global options; native code reads memory buffers too and is no longer. The vendored headers give the exact codes |
| **GIFTI** | **native with pugixml** (new dependency, MIT) | gifticlib is dormant and brings expat and nifti_clib; GIFTI is XML with base64 (optionally gzip) data arrays, short to read and write once an XML parser is there |
| **CIFTI-2** | **native over the NIfTI-2 code and pugixml** | CiftiLib needs Qt or Boost + libxml++ (LGPL). A CIFTI-2 file is a NIfTI-2 file with one XML extension (code 32) describing its rows and columns; the matrix is read by rows |
| **TRX** | **native** (zip read with zlib, which is in the tree; the directory form too) | trx-cpp adds libzip, json11 and mio (memory mapping) to duplicate what we have; its memory mapping does not carry to WebAssembly. A TRX file is a zip (usually stored, not compressed) of raw arrays plus `header.json`. Revisit trx-cpp if TRX's grouping and per-point data outgrow a native reader |
| **XDF** | **native, on demand** | libxdf loads a whole file into memory, which breaks the streaming rule. XDF is a sequence of tagged chunks; a native reader walks them. Its clock synchronisation and de-jittering are methods (pyxdf is their reference) |
| **CTF** (`.ds`: `.res4`, `.meg4`, `.hc`, markers, bad segments) | **native** (no library exists) | big-endian binary with a documented layout; MNE-Python's reader (BSD-3) is the reference and may be ported with attribution |
| **KIT/Yokogawa** (`.con`, `.sqd`, `.mrk`) | **native** (no library) | a binary with a directory of offsets; MNE-Python's reader is the reference |
| **4D/BTi** | **native, on demand** (no library) | the least used; three files (`c,rfDC`, `config`, `hs_file`) and a complex configuration; MNE-Python is the reference |
| **BIDS** | **native** (nlohmann/json, in the tree; a TSV reader) | paths, entities, the inheritance of sidecars, TSV tables. MNE-BIDS (BSD-3) is the reference |
| **MNE `.stc`, `.w`** | **native** | small big-endian binaries. `-stc.h5` (HDF5) later, with section 8 question 6 |
| **FreeSurfer surfaces, annotations; TrackVis `.trk`** | **in progress** in `io/convert/` (not duplicated) | — |

Two new third-party items follow from this: **pugixml** (MIT) and the **NIfTI header definitions** (public domain). Two data files are proposed
(section 8, question 7): MNE-Python's **coil definitions** (`mne/data/coil_def.dat`, BSD-3-Clause), needed to place a FIF, CTF or KIT
sensor's integration points. Each needs the maintainers' OK and a line in `THIRD_PARTY_NOTICES.md`, as `ARCHITECTURE.md` requires.

### 2.3 The parity reference for each format

The reference reads (and where it can, writes) the synthetic test files; its outputs are the goldens. It is a test tool, listed under *Build
and test tools* in `THIRD_PARTY_NOTICES.md`, never a run-time dependency.

| format | reference reader | how the synthetic files are made |
|---|---|---|
| FIF raw, events, annotations, projectors, compensation | MNE-Python (`read_raw_fif`, `get_data`, `annotations`, `info`) | MNE writes them (`RawArray(...).save`), on generated signals and a synthetic `Info` |
| FIF cov, fwd, inv, `-src`, `-trans`, `.stc` | MNE-Python (`read_cov`, `read_forward_solution`, `read_inverse_operator`, `read_source_spaces`, `read_trans`, `read_source_estimate`) | MNE writes them from a synthetic sensor array and a sphere head model on icosphere source spaces |
| EDF/EDF+/BDF | MNE-Python (`read_raw_edf`, `read_raw_bdf`); EDFlib as a second reader | edfio (Apache-2.0) writes EDF; pyedflib (BSD-3-Clause) writes BDF; a generator writes the EDF+D gaps |
| BrainVision | MNE-Python (`read_raw_brainvision`) | pybv (BSD-3-Clause) |
| EEGLAB | MNE-Python (`read_raw_eeglab`) | eeglabio (BSD-3-Clause): both the `.fdt` and the embedded form |
| NIfTI, GIFTI, CIFTI-2 | nibabel (MIT) | nibabel writes them |
| BIDS | MNE-BIDS (`BIDSPath`, `read_raw_bids`, `get_entities_from_fname`) | MNE-BIDS `write_raw_bids` over the synthetic recordings above |
| CTF, KIT, 4D/BTi | MNE-Python (`read_raw_ctf`, `read_raw_kit`, `read_raw_bti`) | nobody writes them: a generator writes minimal synthetic files from the format descriptions; MNE reading them is the check that they are valid |
| XDF | pyxdf (BSD-2-Clause) | a generator writes the chunks |
| TRX | trx-python (BSD-2-Clause) | trx-python writes them |
| MNE `.h5` (`Spectrum`, `CrossSpectralDensity`) | MNE-Python (`read_spectrum`, `read_csd`) | MNE writes them |

Exports are proven the other way round: nxr-neuro writes a synthetic result, the reference reads it, and the test compares the values (and,
where the format has them, the metadata) with what was written.

## 3. The architecture: pure formats, then importers and exporters

Two layers, both in `io/`, so the dependency rule is unchanged (io never includes compute).

### 3.1 Pure readers and writers: `io/formats/` (bytes ↔ in-memory objects)

- **What they are:** functions and small classes that turn bytes into plain C++ objects and back: `nxr::formats::fif`, `::edf`,
  `::brainvision`, `::eeglab`, `::nifti`, `::gifti`, `::cifti`, `::bids`, `::stc`, `::ctf`, `::kit`, `::trx`, `::xdf`.
- **What they never touch:** no `Store`, no `Database`, no subject, no package, no chunk service; no maths beyond decoding (calibration and
  unit scaling are decoding). They do not include `nxr/io/` store, database or writer headers (a new rule in `cmake/check_layering.cmake`).
- **Bytes come from a source:** `ByteSource` (`size()`, `read(offset, n)`) with a file and a memory implementation, and `ByteSink` for
  writers. Memory sources let unit tests build a file in memory and let the WebAssembly client read a file the browser handed it.
- **Recordings** share one shape, the one `BstReader` already has (`io/include/nxr/bst/session.h`):
  - `RecordingHeader`: channels (name, kind, unit, calibration, location and orientation as the file gives them, coil type), sampling
    rate, number of samples, the time of the first sample, annotations (onset, duration, label), bad channels, gaps;
  - `RecordingReader::read(s0, n, c0, c1, out)`: samples `[s0, s0 + n)` of channels `[c0, c1)`, scaled to SI units, row after row, in the
    precision the file holds (32-bit float for integer and float32 formats; float64 for a FIF file stored in doubles).
  - A reader holds the file's directory (FIF tags, EDF record size, CTF trial size) and reads only what a call asks for: memory is bounded
    by the call, never by the recording.
- **Unit-tested now:** round trips (write, read back, equal bytes and values) and the goldens of section 2.3, all on synthetic data.
- They build into their own target, `nxr_io_formats`, which `nxr_io` links, so the wheel, the MEX and the WebAssembly client can use a
  reader without the store (section 8, question 2).

### 3.2 Importers: `io/convert/` (a format's objects → a subject package)

- An importer opens a pure reader and writes **a subject package** (subject-centred spec, section 6: `subject create --package`, then
  `dataset register`), exactly as the Brainstorm importer writes its parts: the same node names, kinds and producer records, so a reader tells
  an imported recording from a Brainstorm one only by its producer.
- **A recording is copied block by block**, as `export_raw_timeseries` copies a BST-BIN file (`io/src/bst_recording.cpp`): blocks of 2^17
  samples (one shard), all channels, through the recording writer and its envelope tee, one block in memory. After that, every computation
  reads the recording through the chunk service (`ChunkSource`, 2^15-sample chunks) like any other. The copy is io's loop, inside io; no
  pipeline sees it.
- **Every item gets its `origin` record:** `source` = the format and the file's path relative to the root the caller named (never an absolute
  path) with the SHA-256 of the bytes read (`fif:sub-01/meg/sub-01_task-rest_meg.fif#sha256=...`); `licence` and `attribution` from the source
  when it states them (a BIDS `dataset_description.json`'s `License`, `Authors`, `HowToAcknowledge`, `DatasetDOI`), otherwise `unknown`;
  `redistributable` true only when the stated licence allows it. The reader's own record (format, version, options, what was skipped) goes into
  the item's `producer_json`.
- **What an importer never copies:** identifying header fields (a FIF `subject_info`, an EDF patient field, names and birth dates in any
  header). Participant variables come in only through `participants import` (subject-centred spec, 3.1). Measurement dates: section 8,
  question 5.
- **The importer rules apply** (section 1): `--parcellation NAME` for any parcellation; `--template NAME@VERSION` or a matching released
  template for anatomy; no default.
- **Command line** (verb first, `import <what>`): `nxr-neuro import fif|edf|bdf|brainvision|eeglab|ctf|kit --from FILE --package PATH
  [--name REC]`, `nxr-neuro import bids --from BIDS_ROOT --dataset PATH [--subjects ...]`, `nxr-neuro import cifti|gifti|nifti ...`, each with
  its options in `cli/command_table.cpp`.

### 3.3 Exporters: `io/export/` (a package → standard files)

- An exporter reads stored nodes by name and hands them to a pure writer. A recording is exported chunk by chunk through the chunk service
  (the exporter states the span: `--from/--to` or `--all`). No maths: an export writes what is stored, in the units stored, or refuses.
- **Provenance travels with every export** (section 5).

### 3.4 Where each part sits in the architecture

| part | layer | may use |
|---|---|---|
| `io/formats/` | data (io), pure | std, zlib, pugixml, matio/HDF5 (EEGLAB), nlohmann/json |
| `io/convert/` importers | data (io) | `io/formats/`, the subject writer, the package and origin API |
| `io/export/` exporters | data (io) | `io/formats/`, the store's readers, the chunk service |
| `cli/` `import`, `export` | orchestration | parses and prints only |

## 4. Per format: what maps to what

Node names follow the Brainstorm importer's (`io/include/nxr/bst/*.h`): a recording Field on channels × time, its `_events` and
`_bad_segments` span Selections, the channel flags partition, the sensor manifold with `_orientation`, `_coils` and `_coil_weights`,
`_headpoints`, the transforms in the sensor manifold's `producer_json["transforms"]`, projectors and CTF compensation as Operators.

### 4.1 FIF (MEG/EEG raw) — read; write later for export

| FIF | subject package |
|---|---|
| `info['sfreq']`, `line_freq`, the acquisition block | the acquisition row and the recording's time line (`t0 = first_samp / sfreq`) |
| `info['chs']`: name, kind, coil type, unit, `cal`, `range`, `loc[0:12]` | the channels, their kinds (the channel-type vocabulary the Brainstorm importer writes; the table is part of the reader's PR), the sensor manifold (`loc[0:3]`), `<ch>_orientation` (`loc[3:12]`), `<ch>_coils` and `<ch>_coil_weights` from the coil definitions (section 8, question 7) |
| `info['bads']` | the channel flags partition |
| `info['dig']` | `<ch>_headpoints` with its label and type Selections |
| `dev_head_t`, `ctf_head_t` | the transforms in the sensor manifold's `producer_json["transforms"]` |
| `info['projs']` | `<ch>_projector_<k>` Operators (kind `projector`), in the file's order; **active** ones are applied to the samples as MNE's `get_data()` applies them, and the recording records which (the Brainstorm importer's rule) |
| `info['comps']` | `<ch>_ctf_compensation` Operator with its grade |
| raw data buffers (`FIFF_DATA_BUFFER`: int16, int32, float32, float64; `cal × range` per channel) | the recording Field, SI units, float32 (float64 when the file stores doubles) |
| `FIFF_DATA_SKIP` | zeros, as MNE fills them, and a span in `<rec>_bad_segments` labelled as a skip |
| annotations; a `-eve.fif` beside it | `<rec>_events`; `BAD_*` annotations to `<rec>_bad_segments` |
| split files (`-1.fif`, …, linked by `FIFF_REF_FILE_NAME`) | one recording |
| `subject_info`, `experimenter`, `description` | not copied (section 3.2) |

**Streaming:** the reader builds the tag directory once and keeps, per data buffer, its first sample and file offset; `read(s0, n, …)` reads
only the buffers that cover the request. Memory: the directory and one buffer.

### 4.2 FIF objects: source spaces, transforms, covariance, forward, inverse; `.stc` — read and write

| object | subject package |
|---|---|
| `-src.fif`, surface source spaces | per hemisphere, the source space's surface (all vertices, in metres, in its coordinate frame) and a Selection of its used vertices (`vertno`); a volume source space as a volume Manifold. The glued cortex is made from them by the existing step, not by the importer |
| `-trans.fif` | the head ↔ MRI transform, in the transforms record |
| `-cov.fif` | `<name>_noise_cov` Operator [M × M] with its channels Selection; `nfree`, method, its projectors and bads in `params_json` |
| `-fwd.fif` | the leadfield Operator (channels ← sources; free or fixed orientation, as stored) with its channels Selection and its source space, as the Brainstorm importer stores a head model |
| `-inv.fif` | the inverse operator's parts as Operators (eigen-fields, eigen-leads, singular values, source covariance, noise covariance, the whitener, the depth and orientation priors) with `params_json.inverse`. **A kernel is not imported:** making one (`lambda2`, dSPM/sLORETA/eLORETA, orientation) is a method, in `meeg/`, with MNE's `apply_inverse` as its golden |
| `-lh.stc`/`-rh.stc` (big-endian float32: `tmin`, `tstep`, vertices, times) and `.w` | a Field on the source space's used vertices; over time, a Field on the surface × a time line, stored as a recording is (the Brainstorm importer's map hooks) |

### 4.3 EDF, EDF+, BDF, BDF+ — read (write on demand)

- Header: per signal label, transducer, physical dimension, physical and digital minimum and maximum, samples per record → channels, their
  units (converted to SI: `uV` → V), calibration from the physical/digital ranges. Signals at different rates are refused unless the caller
  names the rate group to import (each group can be its own recording).
- `EDF Annotations` signals (TAL) → `<rec>_events`; EDF+D/BDF+D gaps → zeros and spans in `<rec>_bad_segments`, never interpolated.
- The patient and recording identification fields are not copied.
- Streaming: records are fixed-size, so `read` seeks to the first record it needs.

### 4.4 BrainVision — read

- `.vhdr`: `DataFile`, `MarkerFile`, `DataFormat` (BINARY only; ASCII refused), `DataOrientation` (MULTIPLEXED or VECTORIZED),
  `BinaryFormat` (INT_16, INT_32, IEEE_FLOAT_32), `SamplingInterval`, `[Channel Infos]` (name, reference, resolution, unit), `[Coordinates]`
  → channels, units, calibration, electrode positions on the sensor manifold.
- `.vmrk`: markers → `<rec>_events`; `New Segment` markers → gaps, as in EDF+D.
- Streaming: multiplexed files by seek to the first sample; vectorized files by one seek per channel.

### 4.5 EEGLAB `.set` (+ `.fdt`) — read

- Through matio: `EEG.nbchan`, `pnts`, `srate`, `xmin`, `chanlocs` (labels, X/Y/Z, type) → channels and positions; `EEG.event` (latency,
  duration, type) → `<rec>_events`; `boundary` events → gaps; `icaweights`, `icasphere`, `icawinv` → Operators recorded, not applied (the data
  in an EEGLAB file are already what the user kept).
- Data: a `.fdt` (float32, channels × samples, sample after sample) streams by seek; data inside the `.set` are read in slabs through matio
  (section 8, question 11 for compressed v5 variables).
- Epoched files (`trials > 1`) are refused in the first slice (section 8, question 10).

### 4.6 CTF `.ds` — read

- `.res4` (big-endian): channels, gains, sensor coils (positions and orientations, number of turns), the compensation coefficients and the
  current grade → channels, `_coils`, `_coil_weights`, `<ch>_ctf_compensation`. A recording whose grade is not the destination grade is
  refused with the reason, as the Brainstorm importer refuses it.
- `.meg4`: int32, trial after trial, each channel contiguous within its trial (the BST-BIN layout): streams by seek.
- `.hc` → the head coils and the head transform; `MarkerFile.mrk` → `<rec>_events`; `bad.segments` → `<rec>_bad_segments`; `BadChannels` →
  the channel flags.
- Continuous recordings stored as several trials become one recording; epoched ones are refused in the first slice.

### 4.7 KIT/Yokogawa — read; 4D/BTi — on demand

- KIT `.con`/`.sqd`: the directory gives the system, channels, sensor geometry, sampling and data offsets; int16 samples, sample-interleaved;
  `.mrk`, `.elp`, `.hsp` → head coils and head points. Streams by seek.
- 4D/BTi: the data file, `config` and `hs_file`; written when a user needs it.

### 4.8 NIfTI-1/2 — read and write

- A T1 (or another volume) → `mri/<name>`: a volume Manifold in closed form with its voxel chart, a Field of intensities, the voxel → world
  transform (sform, else qform) as a 3 × 4 affine, as the Brainstorm MRI importer writes it; a label volume → a codes Selection, only when its
  parcellation is named.
- `.nii.gz` through zlib. Volumes are small: read whole.

### 4.9 GIFTI — read and write

- Read: `.surf.gii` → a surface (pointset + triangles; the winding made canonical by the existing `io/src/surface.cpp`); `.func.gii`/
  `.shape.gii` → Fields on a named surface (refused when its vertex count is not the surface's); `.label.gii` → a parcellation (named only).
- Write: section 5.1.

### 4.10 CIFTI-2 — read and write

- A NIfTI-2 file with one XML extension (code 32): rows and columns mapped as brain models (surface vertices per structure, voxels),
  parcels, scalars, labels or series.
- `dscalar`/`dlabel` on a surface template → Fields and parcellations on that template's surfaces (only when the template package is named
  and its vertex counts match; a CIFTI file on 32k fs_LR needs a released fs_LR template). `dtseries` → a Field on the surface × a time line,
  streamed by rows.
- Write: section 5.1.

### 4.11 BIDS (MEG, EEG) — read

- The layout: `sub-<label>[/ses-<label>]/{meg,eeg,anat}/`, entities parsed from names, sidecars found by the inheritance principle, TSV
  tables.
- `dataset_description.json` → the `origin` record of every item (licence, authors, DOI); `participants.tsv` and `.json` → only through
  `participants import` (never copied by `import bids`).
- One **subject package per `sub-`**; each `ses-`/`task-`/`run-` recording → an acquisition and a recording, read by the format's pure reader
  (FIF, CTF, KIT, BTi, EDF, BDF, BrainVision, EEGLAB).
- `*_channels.tsv` (`status` bad → channel flags; `type`, `units`), `*_events.tsv` (`onset`, `duration`, `trial_type` → `<rec>_events`),
  `*_meg.json`/`*_eeg.json` (`SamplingFrequency`, `PowerLineFrequency` checked against the file), `*_coordsystem.json`, `*_electrodes.tsv`,
  `*_headshape.*` → sensor positions and head points in their stated coordinate system.
- Anatomy: `anat/*_T1w.nii[.gz]` (NIfTI, 4.8); a FreeSurfer subject in `derivatives/freesurfer/` through the FreeSurfer importer (in
  progress). A subject with no MRI gets the anatomy of the template the caller names (the importer rule); none is guessed.

### 4.12 TRX and XDF

- TRX (read and write): `header.json` (dimensions, voxel-to-RAS), `positions.3.<float16|32|64>`, `offsets.<uint32|uint64>`, `dps/`, `dpv/`,
  `groups/` → `curve/<fibers>` exactly as the `.trk` importer (in progress) writes streamlines.
- XDF (read, on demand): regularly sampled EEG streams only; the clock offsets recorded; de-jittering is a method with pyxdf as its reference.

### 4.13 Status

| format | read | write | state |
|---|---|---|---|
| Brainstorm protocols, OBJ/PLY/STL meshes | yes | — | exists |
| FreeSurfer surfaces, annotations; TrackVis `.trk` | yes | — | in progress (`io/convert/`) |
| FIF raw; FIF objects; `.stc` | yes | yes (export) | this spec |
| EDF/BDF, BrainVision, EEGLAB | yes | — | this spec |
| NIfTI, GIFTI, CIFTI-2 | yes | yes | this spec |
| BIDS | yes | derivatives (5.4) | this spec |
| CTF, KIT | yes | — | this spec |
| 4D/BTi, XDF, TRX | yes | TRX | on demand |

## 5. The exporters

Every export carries the same **provenance record**: `nxr-neuro` and its version; the command (verb and options) that made the result, as
`subject show` reports it; the result's `producer_json` (options, span, warnings: what the method cannot resolve); the package's content hash
and the node's key digests; and the `origin` records of its inputs (so a template surface's licence and attribution go with every map drawn on
it). Where a format has room for metadata, the record goes inside the file; it always also goes into a JSON sidecar beside it
(`<file>.json`), so no format loses it. Participant variables are never exported unless the command names them (`--with-participants`).

### 5.1 GIFTI (and CIFTI-2) cortical maps

- **What:** Fields on a cortical surface (a map, a measurement at the vertex level, a statistic per vertex) → `.func.gii` (or `.shape.gii`),
  one per hemisphere (the glued cortex split into its two hemispheres' vertices, `AnatomicalStructurePrimary` = `CortexLeft`/`CortexRight`);
  the surfaces themselves → `.surf.gii`; a parcellation → `.label.gii` with its colour table. A Field over time is not a GIFTI export (use
  FIF/`.stc`, 5.3).
- **Provenance:** file-level `MetaData` entries `nxr-neuro:version`, `nxr-neuro:command`, `nxr-neuro:producer` (JSON), `nxr-neuro:origin` (JSON);
  per data array `Name`, `nxr-neuro:field`, `nxr-neuro:unit` (GIFTI has no unit field); plus the sidecar.
- **CIFTI-2 `dscalar`/`dlabel`:** both hemispheres in one file, for Workbench, on a template whose vertex counts it names. After GIFTI.

### 5.2 TSV measurements and statistics

- **Measurements:** today's `export measurements` (split and wide layouts) gains a BIDS-style JSON sidecar: each column's `Description`,
  `Units` and `Levels`, and the provenance record under `GeneratedBy`.
- **Statistics** (Welch spectra, histograms, summaries stored as statistic tables): long-form TSV, one row per value with its axis
  coordinates (frequency in Hz, bin edges, member), same sidecar.
- The values are what is stored; ratios that are derived at read (a speed, an index) are written as derived columns, named as such in the
  sidecar.

### 5.3 MNE objects

| result | file | MNE reads it with |
|---|---|---|
| a source map, or a source time series, on a source space's vertices | `-lh.stc`/`-rh.stc` | `read_source_estimate` |
| a noise covariance | `-cov.fif` | `read_cov` |
| a leadfield with its source space | `-fwd.fif` | `read_forward_solution` |
| a recording (or a span of it), chunk by chunk | `_raw.fif` (split at 2 GB) | `read_raw_fif` |
| a sensor spectrum; a cross-spectral density | `.h5` in MNE's layout | `read_spectrum`, `read_csd` (section 8, question 6) |

- **Provenance:** FIF files carry a one-line summary in `info['description']` and the full record in the sidecar; `.stc` has no room for
  metadata, so the sidecar holds it.

### 5.4 BIDS-derivatives

- A folder `derivatives/nxr-neuro/` with `dataset_description.json` (`DatasetType: derivative`, `GeneratedBy: [{Name: nxr-neuro, Version,
  CodeURL}]`, `SourceDatasets` from the inputs' `origin` records when they came from a BIDS dataset).
- Per subject (and session), the exports of 5.1–5.3, named with BIDS entities (`sub-`, `ses-`, `task-`, `run-`, `space-<template>`,
  `hemi-L|R`, `desc-<what>`) and each with its sidecar (`Sources` as BIDS URIs when the input was imported from BIDS, plus the provenance
  record). The suffixes for nxr-neuro's measurements are not in the BIDS standard yet (section 8, question 9).
- `participants.tsv` only with `--with-participants`.
- Command: `nxr-neuro export bids-derivatives --dataset PATH --to FOLDER [--subjects ...] --results REGEX`.

## 6. The Python and MATLAB faces: a first slice, aligned with D3

Both bind **the same library API** the command line calls (`pipelines/` and `io/`), so a verb's options have the same names on the command
line, in Python and in MATLAB, and a result made from any face is the same cached result. This needs the dated amendment D3 names: `bindings/`
may depend on `pipelines` and `io`, as `cli/` does (today: `compute` only).

### 6.1 Python (`nxr_neuro`)

```python
import nxr_neuro as nn

ds  = nn.open("study/")                      # a dataset (its catalogue) or one subject package
sub = ds.subject("sub-01")
rec = sub.recording("rest")                  # rec.channels, rec.sfreq, rec.n_samples, rec.units

x, t = rec.read(start=10.0, stop=20.0, channels=rec.channels.of_kind("MEG MAG"))   # NumPy, the span required
for chunk in rec.chunks(all=True):           # the chunk service: one chunk at a time
    ...

r = nn.compute.psd(sub, "rest", all=True)    # = `nxr-neuro compute psd rest --all`; options as keyword arguments
r.cache                                      # "hit" | "computed"
values, axes = r.read()                      # NumPy + named axes, units and the provenance record

nn.ops.welch.psd(x, sfreq=rec.sfreq)         # the pure functions on NumPy arrays (compute/)

nn.import_fif("rest_raw.fif", package="sub-01.nxr.zarr")   # the importers (section 3.2)
nn.export.gifti(r, "out/")                                  # the exporters (section 5)
```

- **First slice** (D3's recommendation): `open`, `subject`, `recording`, `read`, `chunks`; `compute.psd` and `compute.crossspectrum`; a few
  pure functions; `import_fif`, `import_bids`; `export.gifti`, `export.tsv`.
- Errors are exceptions by code (`nn.Error`, with subclasses per `nxr::io::code`); progress is a callback; the C++ releases the GIL while it
  runs; arrays come back without a copy where the stored precision allows it.
- Only NumPy is required. `to_xarray()` and `to_mne()` conveniences work when those packages are installed and are never needed.
- **Build:** nanobind (stable ABI: one wheel per platform for Python ≥ 3.12), scikit-build-core, cibuildwheel in CI: Linux x86-64
  (manylinux) and macOS arm64 first. Versions pinned in `pyproject.toml` (checked 2026-11-16: nanobind 3.1.0, scikit-build-core 1.1.2,
  cibuildwheel 4.3.5). The wheel also installs the `nxr-neuro` command. SQLite, zlib and zstd are linked statically.

### 6.2 MATLAB (`+nxrneuro`, one MEX gateway)

```matlab
ds  = nxrneuro.open('study/');
sub = ds.subject('sub-01');
rec = sub.recording('rest');
[x, t] = rec.read('from', 10, 'to', 20);
r = nxrneuro.compute('psd', sub, 'rest', 'all', true);
[values, axes] = r.read();
```

- Thin `classdef` wrappers over handles held by one MEX gateway that dispatches by name (the pattern `bindings/mex` already has); options as
  name–value pairs with the library's names; errors as identifiers `nxrneuro:<code>`.
- The same first slice as Python. MATLAB ships its own HDF5 and zlib: the MEX hides its symbols (`-fvisibility=hidden`, static linking) so
  they never meet MATLAB's (section 8, question 12).

## 7. The build plan, in PR-sized steps

"Now" means it needs nothing that is not on `main`. Each step is one PR with its tests (scoped, synthetic, through `tools/tests.py`).

| # | step | depends on | when |
|---|---|---|---|
| 1 | `io/formats/` scaffolding: `ByteSource`/`ByteSink` (file, memory), endian helpers, `RecordingHeader`/`RecordingReader`, the `nxr_io_formats` target, the layering rule; `BstReader` made a `RecordingReader` | — | **now** |
| 2 | NIfTI-1/2 pure reader and writer (vendored public-domain headers), nibabel goldens | 1 | **now** |
| 3 | GIFTI pure reader and writer with pugixml (new dependency: the maintainers' OK, notices), nibabel goldens | 1 | **now** |
| 4 | TSV and JSON sidecar writer; `export measurements` gains its sidecar; statistics as long-form TSV | 1 | **now** |
| 5 | FIF tag layer (directory, tags, blocks; read and write) and the raw reader, MNE goldens | 1 | **built**: raw reader and tag reading (`io/formats/include/nxr/formats/fif.h`); tag writing (`fif_write.h`) |
| 6 | EDF/BDF, BrainVision, EEGLAB pure recording readers, MNE goldens (three small PRs, in parallel) | 1 | **now** |
| 7 | BIDS pure layer: entities, sidecar inheritance, TSV tables, MNE-BIDS goldens | 1, 4 | **built** (`io/formats/include/nxr/formats/bids.h`) |
| 8 | FIF objects (source spaces, transforms, cov, fwd, inv) and `.stc`: read and write, MNE goldens | 5 | **built, except `.stc`** (`fif_objects.h`): source spaces, covariances, events, annotation files and transforms read and write (rewritten files equal MNE's byte for byte); forward solutions and inverse operators read; their writers and `.stc` with step 11 |
| 9 | Importers writing packages: FIF raw, EDF/BDF, BrainVision, EEGLAB, with `origin` records; `import <format>` verbs | 5, 6; the package pipelines (open PR) and the `io/convert/` template importers landing | **built as one verb**: `import recording` (`io/convert/recording.cpp`; the format is read from the file's extension; copy only, `--in-place` and a channel type/units table per format to follow) |
| 10 | GIFTI exporter of cortical maps and surfaces; `export gifti` | 3, the package pipelines | after those |
| 11 | MNE exports: `.stc`, `-cov.fif`, `-fwd.fif`, raw FIF chunk by chunk; `export mne` | 8, the package pipelines | after 8 |
| 12 | `import bids` (a package per subject, the importer rules, participants through `participants import`) | 7, 9 | after 9 |
| 13 | BIDS-derivatives exporter | 4, 10, 11 | after 11 |
| 14 | CIFTI-2 read and write (`dscalar`, `dlabel`, `dtseries`) | 2, 3 | **reader built** (`io/formats/include/nxr/formats/cifti.h`); the writer with the exporters |
| 15 | CTF, then KIT readers and importers | 5 (coil definitions), 9 | **pure readers built** (`ctf.h`, `kit.h`); imported by `import recording` (channels and events; CTF coils and compensation operators to follow) |
| 16 | Python wheel, first slice (6.1), with the dependency-rule amendment; cibuildwheel in CI | the amendment (D3) | **now** (in parallel with 1–8) |
| 17 | MEX over the same API, first slice (6.2) | 16 | after 16 |
| 18 | MNE `.h5` (`Spectrum`, CSD), TRX, XDF, 4D/BTi | question 6; demand | on demand |

## 8. Open questions for the maintainers

| # | question | options | recommendation |
|---|---|---|---|
| 1 | Imported recordings: copied into the package, or read in place from the source file | copy block by block (the Brainstorm importer's way) · a linked recording (a `ChunkSource` over a format reader) | **copy**: one reading path for the chunk service, the package's digests and content hash stay true, cache keys stay stable, the package travels alone. Revisit links for very large sources |
| 2 | Where the pure formats live | `io/formats/` with its own target `nxr_io_formats` · a new top-level folder | **`io/formats/`**, own target, a layering rule that it includes no store, database or writer header: no change to the dependency table, and the faces can link it alone |
| 3 | pugixml as a new dependency (MIT; GIFTI, CIFTI-2) | pugixml · expat (MIT, SAX) · a hand-written XML subset | **pugixml**: two files, maintained, builds for WebAssembly; GIFTI and CIFTI files are small enough for a DOM |
| 4 | Units on import | SI (V, T, T/m) as MNE and Brainstorm store them · the file's own units | **SI**, the file's unit and scale kept in the producer record |
| 5 | Measurement dates and identifying header fields | drop · keep · keep only when asked | identifying fields (names, birth dates, patient fields, `subject_info`) **never copied**; the measurement date **kept only with an explicit option**, otherwise times are relative to the recording's start |
| 6 | MNE `Spectrum` and `CrossSpectralDensity` exports, which MNE stores only as HDF5 (`.h5`) | build HDF5 by default in native builds and write MNE's layout · export them as TSV/NumPy only | **HDF5 by default in native builds** (it is already fetched for MATLAB v7.3), not in WebAssembly; after the FIF exports |
| 7 | Sensor integration points for FIF, CTF and KIT (needed by forward modelling) | vendor MNE-Python's `coil_def.dat` (BSD-3-Clause, data) · write the definitions from the papers | **vendor `coil_def.dat`** with its notice: it is MNE's reference, so the forward model's golden stays exact |
| 8 | The `origin` record of user data | keep `OriginRecord` as it is, with the checksum in `source` · add `format` and `checksum` columns | **keep it**: `source` = `<format>:<relative path>#sha256=<hex>`; licence `unknown` and not redistributable unless the source states a licence |
| 9 | BIDS-derivatives names for nxr-neuro's measurements (no standard suffix exists) | `desc-<measure>` with nxr-neuro suffixes, documented · wait for a BIDS extension | **`desc-` and documented suffixes now**, listed in the derivative's `dataset_description.json`; propose them to BIDS later |
| 10 | Epoched sources (EEGLAB epochs, CTF epoched `.ds`, FIF epochs) | refuse in the first slice · concatenate with boundary spans · a new epoch node | **refuse now**, with the reason; design epochs as a Selection of spans later |
| 11 | EEGLAB data embedded in a compressed v5 `.set` | read in slabs through matio if it can · refuse and ask for the `.fdt` form | **check matio's slab reads on compressed variables first**; if it must inflate the whole variable, refuse with a message naming the `.fdt` form |
| 12 | MATLAB's own HDF5 and zlib against the MEX's | hidden symbols and static linking · a MEX without HDF5 | **hidden symbols, static linking**, tested in CI with MathWorks' GitHub actions (free for public repositories) |
| 13 | The name of the pure-function level in Python | `nn.ops` · `nn.functions` · top-level | **`nn.ops`**: `nn.compute.<verb>` already mirrors the command line's `compute` |
| 14 | Priority of 4D/BTi, XDF and TRX | now · on demand | **on demand**, after FIF, EDF/BDF, BrainVision, EEGLAB, BIDS, CTF and KIT |

## Checklist (the architecture policy)

- Can each new compute function be called with a matrix from anywhere? No compute function is added; kernel assembly from an imported MNE
  inverse (4.2) is a `meeg/` function of matrices with MNE's golden.
- Can the pipeline run the same chain on another input of the same kind just by naming it? Imported recordings are ordinary recordings: every
  verb runs on them by name.
- Is every loop over samples or chunks, and every file access, inside io? Yes: pure readers, importers and exporters are all in `io/`; the
  import copy is io's block loop, exports of recordings use the chunk service.
- Is every choice of which data (signal, source, span) made by the caller? Yes: the files, the parcellations, the template and an export's span
  are named by the caller; nothing is defaulted.
