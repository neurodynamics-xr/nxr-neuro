# io — the nxr dataset format (nxr_io)

This folder is the **io library** of nxr-neuro (the former nxr-io repository, imported with its history). It is `nxr_io`: the nxr dataset format
(SQLite rows + Zarr v3), its reader and writer, and the Brainstorm importer (`nxr::bst`). Its dependencies: SQLite, the Zarr codecs (zstd, blosc, zlib),
matio and HDF5. It never includes `nxr_compute`; the pipeline layer (`../pipelines`) is where they meet. The command line is `../cli`
(`nxr-neuro`), the one command for every verb below. Paths in this document are relative to this folder (`src/`, `include/`, `schema/`, `test/`);
the build is the top-level one (`cmake -S . -B build -G Ninja` at the root of the repository, see `../README.md`).

The TypeScript package of the old repository (`@nxr/io`) was removed here; it stays in the history and in the nxr-io repository.

## Step 1: reading

The C++ core (`src/`, `include/`) reads an nxr dataset the way nxr's Python does: `dataset.sqlite` at schema 50 (SQLite, read-only; other versions
are refused with nxr's wording), and the Zarr v3 arrays its rows name (single chunk or regular grid, chunk key encoding `default` with separator `.` or `/` (the array's own zarr.json) or `v2`, `sharding_indexed` read through
the shard index, codecs `bytes` · `blosc` · `zstd` · `gzip` · `crc32c`; a missing chunk is the fill value; regions are read chunk by chunk).
`nxr-neuro read <dataset folder>` lists the subjects and every array the rows name (shape, dtype, CRC-32 of the decoded bytes);
`--array <subject> <path> [--region a:b,c:d] [--out FILE]` reads one array or region.

Build (CMake + Ninja; c-blosc 1.x, matio and HDF5 are fetched into the build directory, SQLite and zlib are the system's):

    cmake -S .. -B ../build -G Ninja && cmake --build ../build

Parity with nxr's Python (decoded bytes of every array equal `zarr.open_array(...)[...]`; nxr's fixtures plus a dataset generated
with nxr's own writer; synthetic data only):

    export NXR=<path to the nxr checkout> NXR_NEURO=$PWD/../build/cli/nxr-neuro NXR_NEURO_SCENARIO=$PWD/../build/io/nxr-neuro-scenario
    uv run --project $NXR python -m pytest -q test/parity/test_parity.py

## Step 2: writing

The C++ core WRITES an nxr dataset the way nxr's Python does (`nxr.io.writer`, `database`, `writer_lock`, `layout`), organised by the same
nouns: **the same rows, the same `zarr.json` bytes, the same arrays** (the compressed chunk bytes too), and the two lock, read and recover each
other's datasets. The format is nxr's; nothing is invented (`schema/model.sql` is a byte copy of the Studio's schema, embedded at build).

**The procedure** (D59/D141): bytes first, the row last. CREATE = the rows (`pending`) -> LAY OUT (the node's `zarr.json` is its row: the schema's
view text, byte for byte, written as JavaScript's `JSON.stringify` writes it) -> POPULATE (the bytes) -> COMPLETE. A failed composition rolls back what
it created (`abort`, `rollback_to`). DELETE removes only the arrays no remaining row names (the Studio's rule, D164). RECOVER at open, only by the process
that took the writer lock. One writer a dataset: `<dataset>/.writer.lock`, byte-compatible with nxr's and the Studio's.

**API** (C++17, namespace `nxr::io`; `#include "nxr/io.h"` for all of it), the names are nxr's:

| nxr (Python) | nxr-io (C++) | header |
|---|---|---|
| `create_dataset(datastore, name, ...)`, `open_writer(folder)`, `delete_dataset`, `read_catalog`/`write_catalog` | same names; `DatasetWriter` (RAII: closes and releases the lock), options structs `CreateOptions`, `OpenOptions`; `list_datasets` | `writer.h` |
| `SubjectWriter.create / .open` | `SubjectWriter::create(ds, name, {.kind = ...})`, `::open(ds, name)` | `writer.h` |
| `.manifold / .field / .operator / .selection / .dictionary / .acquisition / .product / .topology / .geometry` | `manifold(ManifoldSpec)`, `field(FieldSpec)`, `operator_(OperatorSpec)`, `selection(SelectionSpec)`, `dictionary(DictionarySpec)`, `acquisition(label)`, `product(factors)`, `topology(...)`, `geometry(...)`, `measurement(selection_id, row)`; any column not a field of a spec goes in its `cols` (`Cols().set("n_faces", 80).json("producer_json", Json{...})`) | `writer.h` |
| `complete()`, `abort()`, `mark()`, `rollback_to()`, `composition()` | same; `compose(body)` completes on return, aborts on an exception | `writer.h` |
| `create_node / update_node / remove_node / recover` | same | `writer.h` |
| `write_array`, `write_sparse`, `array_meta`, `default_chunks`, `layout_array`, `populate_array` | same; an array in is `view(vector_or_pointer, {shape})`; `ArrayOptions{.chunks, .compress, .shards, .dense}`; the policy (format 51, 2026-11-14): one chunk up to 4 MiB; above it axis 0 in ~2 MiB inner chunks, SHARDED by default in shards of up to ~64 MiB (`kShardBytes`; `.unsharded` opts out; `.in_pieces` = unsharded AND recorded in zarr.json as `nxr_written_in_pieces`, for an array written region by region, so `compact` keeps it unsharded), blosc zstd level 3, byte shuffle, typesize = itemsize; shards as given; inner chunks in morton order; a chunk equal to the fill is not stored unless `dense`) | `array_writer.h` |
| `open_database`, `drain`, `resync_all`, `text_parity`, `uuid7`, `now_utc`, `js_number`/`js_json`/`zarr_json_text` | `Database::open`, `drain`, `resync_all`, `text_parity`, `uuid7`, `now_utc`, `js_number`/`js_json`/`zarr_json_text` | `database.h`, `json_text.h` |
| `acquire_writer_lock`, `read_writer_lock`, `NXR_WRITER_LOCK_HELD` | same | `writer_lock.h` |
| `folder_of / node_path / folder_for / safe_name / atlas_node`, `coordinate_system` | same | `layout.h`, `writer.h` |
| `nxr.io.mesh.read_mesh` (OBJ, PLY, STL), `add_surface`, `canonicalize_winding`, `check_mesh` | same | `surface.h` |
| exceptions | one type, `nxr::io::Error`, with a stable `code()` (`invalid_argument`, `already_exists`, `not_found`, `no_such_row`, `writer_lock_held`, `not_writable`, `schema_version`, ...) and nxr's wording in `what()`; `python_name()` is the exception nxr raises | `errors.h` |

    auto ds  = create_dataset(datastore, "demo", {.created_by = "me"});
    auto sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      auto m = add_surface(sub, {.name = "sphere"}, vertices, faces).id;          // or sub.manifold({...})
      sub.field({.name = "f", .path = "source/f", .kind = "map", .manifold_id = m, .data = view(values, {n})});
    });                                                                               // complete on return; abort on an exception

**Commands** (`nxr-io`; JSON lines on stdout, `{"stage": "result", ...}` / `{"stage": "error", "message": "<Python exception>: ..."}`, exit 0 / 1, 2 for usage):

    nxr-neuro dataset create --datastore D --name N [--source-tool T --source-protocol P --source-path S] | list --datastore D | show --dataset D/N
    nxr-neuro dataset create … --templates FS_SUBJECTS [--default fsaverage5] [--modes K]   with the default subject (below)
                  | delete --datastore D --name N | recover --dataset D/N
    nxr-neuro subject create --dataset D/N --name S [--kind subject|template|group] | list | show --subject S | delete --subject S [--node PATH]
    nxr-neuro surface add --dataset D/N --subject S FILE [--name N] [--scale F] [--primary | --no-primary]      # OBJ, PLY, STL
    nxr-neuro read <dataset folder> [--array <subject> <path> [--region a:b,c:d] [--out FILE]]                    # step 1's reader

Not in this step: GIfTI and FreeSurfer surfaces (nxr reads them through nibabel), `nxr subject copy`, the importers, bindings.

**Parity with nxr's Python** (`test/parity/test_write_parity.py`; synthetic data only): the same scenario (dataset -> subject -> surface from a generated
icosphere -> small, blosc, sharded and dense fields, a sparse field, a dictionary and a selection coded by it with a measurement, an acquisition, a time
line and a product, an operator) is written through nxr's Python and through the C++ API (`nxr-io-scenario`), both with a counter for ids and a fixed
clock; every row of every table, every `zarr.json`, every chunk file and the decoded arrays are compared. Also: cross-reads (`nxr.io.open`,
`text_parity`, the `show`/`list` commands), the writer lock in both directions, recovery of an interrupted write, the delete rule, each command's output
against nxr's, the mesh formats, `js_number`.

    cmake -S .. -B ../build -G Ninja && cmake --build ../build && (cd build && ctest)
    export NXR=<path to the nxr checkout> NXR_NEURO=$PWD/../build/cli/nxr-neuro NXR_NEURO_SCENARIO=$PWD/../build/io/nxr-neuro-scenario
    uv run --project $NXR python -m pytest -q test/parity

The delete rule (an array another row names stays, D164/D167) is the same in nxr 1.7.0 and nxr-io, and `delete_dataset` refuses a name with a path in it in both.

## Step 3: importing from Brainstorm

The C++ core imports a Brainstorm protocol the way nxr's Python does (`nxr.convert`), one piece at a time. Step 3 (so far) ports the reading of
Brainstorm's `.mat` files, the protocol listing and scan, **a subject's anatomy**, **the sensors with the full geometry of register D166 and the channel file's projectors and CTF compensation (D168)**, **the subject's MRI**, **the sessions** (recordings, flags, events, bad segments, resample), **the inverse kernels and head models**, **static maps**, **source power by band**, **fibres and connectomes**, **the template and group subjects**. The verbs are nxr's: `nxr-neuro convert …`
is `nxr convert …`, with the same JSON lines.

**Reading `.mat` files: `nxr::bst`** (`#include "nxr/bst.h"`; `include/nxr/bst/mat.h`). [matio](https://github.com/tbeu/matio) (BSD, pinned release
`v1.5.28`) is fetched and built inside the build directory by CMake, with zlib **and HDF5**; nothing is installed. A `.mat` is opened
(`MatFile::open`), its variables listed (`list()`: name, class, size), and one variable read as a typed `Value` (`read(name)`; `read_variable(file, name)`;
`read_variables(file, names)` is pymatreader's `variable_names`): numeric arrays of any class (real or complex, in the stored type, widened on demand:
`doubles()`, `int64s()`, `row_major()`), logical, char (UTF-8), sparse, **structs and struct arrays with their elements**, cells, empties, exactly as stored.
MATLAB v4, v5, v6, v7 (compressed) **and v7.3 (HDF5)** files are read, and a v7.3 file reads as its v7 twin: same classes, sizes, values, struct arrays,
cells, sparse and complex parts, empties, text (`MatFile::version()` says 73). Brainstorm writes v7.3 only for a file above 2 GB uncompressed (`bst_save.m`).
One thing a v7.3 file cannot give: a MATLAB *object* (string, datetime, table, categorical, a user class) is stored there as an opaque reference array and a
hidden class table; `read(name)` of a variable holding one throws `NotImplementedError: <file>: variable '<name>' holds a MATLAB object ... stored opaquely` (and
`read_all` returns it as `Class::Opaque`). Brainstorm's files hold plain structs, cells, char and numbers.

**HDF5** (the official [HDFGroup/hdf5](https://github.com/HDFGroup/hdf5) release `hdf5_1.14.6`, which matio 1.5.28 supports: it asks for 1.8 to 1.14) is fetched by
`FetchContent` and built static inside the build directory: C library only (no Fortran, C++, high-level library, tools, examples or tests), not thread-safe,
the zlib filter on (the system zlib nxr-io already links), no szip. A cold configure adds a shallow clone (about 30 MB) and the build adds about 7 s of compile
on a 10-core Mac (`ninja hdf5-static`); matio's CMake is pointed at that build by `cmake/patch_matio_hdf5.cmake`. Nothing is installed.
**For bindings (MEX, Python extension):** HDF5 exports its API whatever the visibility preset, so a shared object that links nxr-io would export `H5*` and
clash with the HDF5 that MATLAB or h5py loads. Call `nxr_io_hide_static_symbols(<target>)` (in `CMakeLists.txt`) on that target: `-load_hidden` of the HDF5
and matio archives on macOS (checked: a dylib linking nxr-io exports no `_H5*`/`_Mat_*` symbol with it), `--exclude-libs,ALL` on Linux (the documented flag,
not yet tried here). Nothing needs doing for the `nxr-io` executable.

**Commands** (`nxr-neuro convert …`; JSON lines on stdout, `{"stage": "error", "message": "<Python exception>: ..."}` and exit 1 on failure):

    nxr-neuro convert list <protocol>                                    what the protocol holds, from its folders (nxr convert list)
    nxr-neuro convert scan <protocol> [--prefer protocol.mat|walk] [--summary] [--out FILE]
                                                                      the protocol as a source view, `nxr.source@1.0` (nxr convert scan)
    nxr-neuro convert subject <protocol> --subject S --dataset <datastore>/<dataset> [--resample HZ]
                                                                      the whole subject as ONE composition (below)
    nxr-neuro convert convert <protocol> --subject S --condition C --dataset D [--surface F] [--extra-surface F ...] [--recording data_X.mat] [--channel-file F]
                  [--no-gain] [--all-surfaces] [--include-imported] [--no-raw] [--resample HZ]     one condition as a session (nxr convert convert)
    nxr-neuro convert resample --dataset D --subject S --recording NAME --to HZ [--name N]               a stored recording resampled beside it
    nxr-neuro convert badsegments --dataset D --subject S --from JSON [--replace]                         bad segments detected after the conversion
    nxr-neuro convert events --dataset D --subject S --from <protocol data folder> [--replace]            a subject's Brainstorm events onto its recordings
    nxr-neuro convert template <protocol> --dataset D [--surface tess_*.mat …]
                                                                      anat/@default_subject as the TEMPLATE subject (D55)
    nxr-neuro convert group <protocol> --dataset D                       data/Group_analysis/@intra as the GROUP subject (D55)
    nxr-neuro convert templates <references-fsaverage-ico7> --dataset D [--subject fsaverage] [--trk F] [--surface NODE]
                                                                      the templates-and-references dataset (needs the atlas port, below)
    nxr-neuro convert mri --dataset D --subject S --anat <anat dir> [--only N ...]
                                                                      the subject's MRI volumes (nxr convert mri)
    nxr-neuro convert surface --dataset D --subject S --file tess_*.mat … [--bst-root P]
    nxr-neuro convert inverse --dataset D --subject S --file results_*KERNEL*.mat --acquisition C --bst-root P [--channels-id I] [--surface-id I] [--forward-id I] [--name N] [--channel-file F]
    nxr-neuro convert map --dataset D --subject S --file results_*.mat|matrix_*.mat … --bst-root P [--acquisition C] [--replace] [--provenance FILE]
                                                                      add Brainstorm surfaces to a subject (nxr convert surface --file)

**What is ported.** `nxr.convert.protocol` (the listing), `nxr.convert.sources` (the scan: `data/protocol.mat` with its surfaces, `iCortex`, kernels' own
`SurfaceFile`, raw links' format and duration, importability; the walk when there is no index), `nxr.convert.naming` (node names from file stems),
`nxr.convert.surface.export_surface` and `add_surface` (vertices and faces, winding canonicalisation, mesh health, the hemispheres from the Structures atlas, the
registered sphere `Reg.Sphere` as a second geometry of the same topology, the parcellations as a set of partitions coded by app-wide dictionaries, which surface
is primary) and, of `nxr.convert.subject`, the anatomy folder (`UseDefaultAnat` borrows `@default_subject`) and the choice of the primary surface (the kernels'
`SurfaceFile`, else `iCortex`, else `tess_cortex_pial_low.mat`). `add_surface` (`surface.h`) takes the sphere, the hemispheres and the parcellations as options.

**Projectors and CTF compensation** (`#include "nxr/bst/projectors.h"`, `bst::export_projectors` / `export_compensation`; `nxr.convert.projectors`, register D168). `ChannelMat.Projector` becomes Operators on the channels, `<ch>_projector_<k>` (dense [M x M] f64; params status · comment · method · kind ssp|ica · components), `MegRefCoef` the operator `<ch>_ctf_compensation` (csr [M x M], entry MEG x MEG REF; params grade). SSP: P = I - U U^T; ICA: W = Components^T, P = I - pinv(W)[:, removed] W[removed, :]. The session import APPLIES the status-1 projectors to the samples it copies (`bst::Projection`; status 2 recorded as already applied, 0 stored only) and records which on the recording's provenance. The arithmetic is fixed (ascending sums, products not fused, Gauss-Jordan pinv) so the bytes equal nxr's: **`test/parity/test_projectors_parity.py`**.

**Sensors** (`#include "nxr/bst/sensors.h"`, `bst::export_sensors`; `nxr.convert.sensors.export_sensors`). The sensor manifold is one point per channel (`Loc(:,1)`; a
bipolar electrode channel at the midpoint of its two contacts, with `_contact1/2`), its names and montage families as Selections coded by dictionaries
(`bst/entities.h`: `labelling`, `label_set`, `flag_levels`, as `nxr.convert.entities`). Register D166 adds, all in `sensor/`: Field `<ch>_orientation` (`Orient(:,1)`),
Manifold `<ch>_coils` (every `Loc` column of every non-bipolar channel) with Field `_coils_orientation` and Operator `<ch>_coil_weights` (csr [M x P], coils -> sensors,
`Weight`; at the subject's root, the layout rule), Manifold `<ch>_headpoints` with its `label` and `type` Selections, and the transforms (`TransfMeg`, `TransfEeg` with
their labels, the `SCS` fiducials) in the sensor manifold's `producer_json["transforms"]`. The session import calls it (below); the tests reach it through `convert convert`.

**The MRI** (`bst/mri.h`; `nxr.convert.mri`): `subjectimage_*.mat` -> `mri/<name>`: the T1 (volume manifold in closed form with its voxel chart, the intensities as a dense
field in cubic 64-voxel chunks, the voxel -> SCS and -> NCS affines as `transition` operators), an atlas volume as CODES (a Selection on the T1's volume read with an app-wide
dictionary; Brainstorm's label value kept per entry, values the table does not name get an `unlabelled <v>` row) and a float map (a Field on the T1's volume). A left-right
mirror (LPA must map to +y, RPA to -y) is refused, as is a frame or grid that is not the T1's; an unreadable image is reported (`stage: skipped`) and the rest imported; none
imported is a FileNotFoundError. `convert subject` imports the images after the surface (nxr does it after the sessions) and `convert mri` adds them to an existing subject.

**Sessions** (`bst/session.h`; `nxr.convert.condition`, `.raw`, `.timeseries`, `.events`, `nxr.io.grid`). `convert convert` mirrors one condition as a session (an acquisition with its BIDS
entities, D161): the anatomy if absent, the channels (shared with an earlier session when the file and flags are byte-identical, else qualified, D121; sha1 fingerprint), then the
raw link's recording and/or the imported `data_*.mat` excerpts. **COPY, NEVER LINK**: samples are read from the `.bst` (BST-BIN: `hdrsize`, epochs of [nchannels x epochsize] float32; header from the
link's `F.header`; each channel's block is `epochsize` samples apart in EVERY epoch, the last one included, as Brainstorm writes and reads it (`out_fwrite_bst.m`, `in_fread_bst.m`): the file ends after the last channel's declared samples, or holds the last epoch whole, and only the declared samples are kept; a last epoch re-laid with its channels their declared length apart (`cut`) is read too (issue #276); a size that fits none of these layouts is refused, naming them; a binary CUT SHORT is read in Brainstorm's layout, keeps the samples every channel has in full and reports `truncated` {file, declared_samples, read_samples, seconds_lost} in the progress and the result, one under half the declared samples is refused) or the `.mat`'s `F` (read by column slab through matio) and written into the dataset's own Zarr array. A moved binary is relinked beside the link (`stage: relinked`, D122);
a link of another format (CTF `.ds`, FIF, EDF ...) is refused with nxr's message. **The timeseries grid** (nxr's `io/grid.py`, not the general array policy): raw [channels x samples] float32,
inner chunk [1 x 2^15], shard [channels x 2^17], UNCOMPRESSED (clipped for a recording shorter than a chunk); the ENVELOPE `<rec>_envelope` [channels x windows x 2] (min, max per 2^5 samples),
inner [1 x 2^12 x 2], shard [channels x 2^12 x 2], reduced from each block as it is written (one pass); the `<rec>_chunks` and `<rec>_envelope` spans Selections state both grids in closed form with
the envelope's array-backed min/max measurements (D46/D57/D96). Writing is one 2^17-sample block (all channels) at a time (`SlabWriter`, which streams a shard chunk by chunk to its file), so memory
is bounded by about two blocks, ~1 MB per channel (an excerpt's `.mat` is read a block at a time; a resample holds the resampled recording whole, as nxr does). Also: the channel flags
(`sensor/<rec>_flags`), the events (`time/<rec>_events`, `_bad_segments`, per-occurrence channels in `params_json`), `badsegments` and `events` onto an existing recording and every recording derived from
it, and `resample` (`scipy.signal.resample_poly` ported: Kaiser-window `firwin`, polyphase `upfirdn` in float64; the sample-count rule and refusals as nxr's). `convert subject` converts every session.

**Inverse kernels, head models, maps, source power** (a straight translation of the Python, not yet compared with it: one round of parity tests comes after the whole
port is integrated; built warning-clean and smoke-run on a tiny synthetic protocol). `bst/inverse.h` (`nxr.convert.inverse`; `add_inverse` of `nxr.convert.condition`): `export_inverse`
(the kernel as an Operator [nsrc, M] with its method recorded in `params_json.inverse`, paired with its leadfield by `inverts_id` when the ends swap; `_channels`, `_noise_cov`,
`_whitener`, `_source_rr`/`_source_nn`), `export_forward` (the head model, declared without bytes when there is no gain, with `_GridLoc`/`_GridOrient`), `inverse_method_of`
and `add_inverse` (`nxr-neuro convert inverse`: references found in the subject, the name Brainstorm's stem, session-qualified if taken, never overwriting). The `--from`
carry-inverse logic is gone in nxr and is not translated. `bst/maps.h` (`nxr.convert.results_map`; `nxr-neuro convert map`): source maps (a Field on the surface the map's own
`SurfaceFile` names, exported beside the primary if absent), region maps (measurements on the atlas's Selection) and `template_map_files`; `bst/timefreq.h` (`nxr.convert.timefreq`)
and `bst/frequency.h` (`nxr.convert.frequency`): source power by band, one Field per band, the band a span of the default subject's Frequency manifold.
**How they are connected** (the integration of the three ports). `convert_condition` calls `import_condition_kernels` after the recordings and `export_condition_timefreqs` after the surfaces,
exactly where `nxr.convert.condition` does. A map over time and a confounds matrix are written by the recording writer (`bst::recording_map_hooks()`: `MapHooks::series` is `_recording` with `_write_blocks`,
`MapHooks::time_line` is `_time_line`), so they import (`convert map`, `convert subject`). The names of session-scoped nodes (`bst/naming.h`: `claim_node_name`, `source_fingerprint`, `kernel_node_name`,
`headmodel_node_name`, `acquisition_label`, `recording_node_name`, `channel_node_name`) are one set, shared by sessions and source. `ensure_band_dictionary` is the writer's (`nxr/io/writer.h`, as `nxr.io.writer`'s),
and `frequency` makes the dictionary when a dataset lacks it. There is one `ConditionPlan` / `condition_plan` (`bst/subject.h`, `nxr.convert.subject`).

**Fibres** (`bst/fibers.h`; `nxr.convert.fibers`): a `tess_fibers_*.mat` (`Points` [nFibers, nPoints, 3], no `Vertices`) as a `curve` manifold in `fibre/` (path topology,
fibre-major points in SCS metres, the colours when Brainstorm has them), the operator `<fibers>_ends` (csr, each fibre's first and last point to its nearest cortex vertex, one 1 per
end), the connectome `source-source/<fibers>_connectome` (a sparse CSC field on cortex x cortex, D159: W = E1^T E2 + E2^T E1 from the ends) and Brainstorm's own region-level
connectomes (`Scouts(k).Assignment`, per atlas) as `streamlines` measurements on the atlas Selection (D158). Reached through `convert subject`.

**The whole subject** (`bst/subject.h`; `nxr.convert.subject`): `convert subject` writes every part of one subject as ONE composition, all or nothing, in nxr's order and with
nxr's lines: 1. each condition with recordings or a raw link, sorted (a session each; the first brings the anatomy; one that cannot convert is rolled back and reported as skipped),
2. the subject images, 3. the source maps of every condition folder, 4. the fibres on the subject's cortex (`iCortex`, else `brainstormsubject.mat`'s `Cortex`, else the primary surface).
As nxr, a subject with no importable condition is refused. `condition_plan` (the surface of the kernels, the raw link, the channel file, the kernels) is `nxr.convert.subject.condition_plan`.
`convert template` (`anat/@default_subject`: every low-resolution cortex, pial first and primary; its images; the static maps on it) and `convert group` (`timefreq_*` of
`data/Group_analysis/@intra`) are `convert_template` and `convert_group`. `convert templates` is `nxr.convert.templates.build` (fsaverage and the reference maps); its readers (FreeSurfer
surfaces and labels, GIfTI) and `DefaultSubjectWriter` belong to the atlas port.

**The default subject** (`include/nxr/io/default_subject.h`; `dataset create --templates`): `@default_subject` (kind `template`), written as any subject is (`add_surface`, one
composition): `cortex_pial` (primary; its hemispheres `_1` left, `_2` right; `cortex_pial_sphere` = `sphere.reg` as stored), `cortex_white`, `cortex_inflated`, read from
`<templates>/<default>/surf/{lh,rh}.*` (`read_freesurfer_surface`, nibabel's `read_geometry`), mm to m, coincident vertices separated (nxr's `separate_coincident`), and the pial's
parcellations `source/cortex_pial_atlases` (Structures, Desikan-Killiany, Destrieux from `label/{lh,rh}.{aparc,aparc.a2009s}.annot`: `read_freesurfer_annot`, nibabel's `read_annot`;
nxr's `DefaultSubjectWriter.parcellations`: `unknown`, `Medial_wall`, `corpuscallosum` and empty regions left unassigned, a family missing a hemisphere left out). With `--modes K`
(pipelines built), `dataset create` then solves the pial's Hodge–Dirac basis with `eigen solve`'s pipeline: io writes no modes, as the solve is compute's. The dataset and its default subject are made together: a failure removes the
dataset the command made. Parity: `test/parity/test_default_subject_parity.py` (the surfaces against
nibabel; the parcellations against nxr's `parcellations`). The rest of nxr's default subject (group tree, frames, option resolutions, HCP-1065, the Frequency manifold) is not written yet.

**Not implemented** (stated, not stubbed): the rest of the atlas/nibabel layer that `convert templates` (the template-folder BUILDER, not a Brainstorm import) needs:
`read_label` (FreeSurfer labels), `read_gifti`, `default_surface` and `default_parcellations` (`DefaultSubjectWriter`) and `hcp_fibres` (the HCP-1065 `.trk`). They are declared in `bst/templates.h` and each throws
`NotImplementedError` naming the missing reader (end of `src/bst_templates.cpp`); `convert templates` therefore stops at its first map (GIfTI). (`read_hemis`, FreeSurfer surfaces, is implemented.) The other commands (`qc`, `mne-eeg`, `mne-meg`, `bids-ieeg`, `migrate`,
`fragment`, `merge`, `atlas`) answer `NotImplementedError: nxr-neuro convert X is not ported yet`; `surface --fs-dir` (FreeSurfer) is not an option yet. `convert subject` refuses a subject with no importable
condition, as nxr does ("nothing to anchor a subject").

**Three import fixes shared with nxr 1.8.0** (`test/parity/test_bst_import_fixes_parity.py`): `convert inverse` refuses a kernel whose stem is already in the same acquisition
(the acquisition id is resolved to its label before it is compared); an unreadable `subjectimage_*.mat` (a v4/v5 file that ends inside a variable, as one that matio would
read only in part) is a `skipped` line, `<file>: not a readable MATLAB file`, and the rest of the subject imports; `producer_json.source_orient` is always a list of strings
(a char or a 1x1 cell -> `[it]`, a cell -> the list, empty -> `[]`). Merge nxr first: the parity tests compare against nxr main.

**Parity with nxr's Python** (`test/parity/test_convert_parity.py`; synthetic data only: octahedra and icospheres written with `scipy.io.savemat` by nxr's own
test helpers `tests/convert/eeg_fixture.py` and `atlas_meshes.py`). Each scenario runs through `nxr convert …` and through `nxr-neuro convert …` (ids a counter and a fixed
clock on both sides: `NXR_IO_DETERMINISTIC=<n>` for nxr-io, a test-only switch) into two copies of one starting dataset; every row of every table (ids included), every
`zarr.json` as text, every chunk file's bytes and the printed lines are equal. Scenarios: a cortex with two hemispheres, atlases, a registered sphere (v7 compressed
and v6 uncompressed), a plain octahedron, a mis-wound hemisphere (winding flipped), a head surface (`mri/`), an unhealthy mesh (warned, imported), unicode labels, a
Structures atlas with one hemisphere, two cortices sharing an atlas dictionary, a failed import (rolled back whole), `convert subject` on a protocol with sessions,
on one that borrows the default anatomy and on one without sessions; **`test/parity/test_session_parity.py`** (the sessions: the grid and envelope at 1000, 32768, 40000, 135072 and 299 thousand samples with zarr.json and shard bytes, flags, events, imported excerpts, relinking, refusals, resample, bad segments, events onto recordings), **`test/parity/test_sensors_mri_parity.py`**: the sensors (MEG with two coils of four integration points and weights, a MEG REF, EEG single points, a bipolar sEEG pair with its shaft, head points with labels, the transforms, flags) and the MRI (a 16^3 T1 with its NCS, an atlas with a value the table does not name, an atlas of 300 labels (uint16), a float map named by its comment, v6 and v7, `--only`, a mirrored volume, a grid that is not the T1's, an int16 volume, a v7.3-shaped HDF5 file without a Cube, a subject with surfaces and images), every row, `zarr.json` text and chunk byte equal; `list` and `scan` (index and walk, v6 and v7 index, `--summary`, `--out`, `--prefer`). v7.3 is
read through HDF5 and matio and checked against nxr's pymatreader by the first parity round (round 1, below); `ctest` runs `nxr-io-bst-compare` on genuine MATLAB R2023b files (`test/data/mat-v73/`: a Brainstorm-like
tess with atlas and scout struct arrays and `Reg.Sphere`, sparse real and complex, cells of strings and nested cells, logical, complex, int8 to uint64, N-d, empties, a
character beyond the BMP) saved `-v7.3` and `-v7` by `make_v73_smoke.m`, and requires each v7.3 variable equal to its v7 twin, class by class and byte by byte;
`test_convert_parity.py` imports a MATLAB-converted v7.3 surface and scans/imports a v7.3 kernel (they need MATLAB, `NXR_MATLAB`, and skip without it);
`test/parity/test_bst_reading.py` reads files scipy wrote (every class,
complex, logical, unicode, struct arrays, cells, sparse, v4/v6/v7) through `nxr-io-bst-dump`; `test_bst` (ctest) does it with files matio wrote.

    cmake -S .. -B ../build -G Ninja && cmake --build ../build && (cd build && ctest)
    export NXR=<path to the nxr checkout> NXR_NEURO=$PWD/../build/cli/nxr-neuro NXR_NEURO_SCENARIO=$PWD/../build/io/nxr-neuro-scenario NXR_NEURO_BST_DUMP=$PWD/../build/io/nxr-neuro-bst-dump
    uv run --project $NXR python -m pytest -q test/parity

**Where it differs from nxr's Python, on purpose** (matio returns what is stored; pymatreader squeezes it): a struct array is read element by element (nxr
re-assembles pymatreader's struct-of-arrays with heuristics, `_records`, `struct_rows`, `_as_list`), so none of those traps exists here; an empty char is an empty
char (pymatreader returns an empty array, and nxr then raises "truth value of an empty array is ambiguous" on a surface whose `Comment` is empty; nxr-io imports it
with no comment); a one-cell array (Brainstorm's `Condition`) is read as its cell, as pymatreader squeezes it. A v7.3 kernel is read like a v7 one (its SurfaceFile names the surface); an HDF5 file with MATLAB's header but no MATLAB variables is, in a scan, unknown (no format, no surface). `nxr convert list` and `scan` of a folder that is no protocol end in a traceback; nxr-io prints
the same message as a `stage: error` line.

**Against a real protocol** (optional; set `NXR_IO_BST_PROTOCOL` to a Brainstorm protocol, `NXR_IO_BST_SUBJECT` to ONE subject of it, `NXR_IO_BST_OUT` to a folder for the two
datasets, and `NXR_IO_BST_CHANNEL` to one channel file; read only): `test_a_real_subject_anatomy_and_images_are_nxrs` and `test_a_real_channel_file_is_nxrs` run the same comparison
on the subject's own anatomy folder (its surfaces and images; never `convert subject`, which lists the whole protocol) and that channel file. Unset, they skip.

## Pipelines (nxr_io x nxr_compute)

nxr_io reads and writes datasets; `nxr_compute` (`../compute`) does the geometry and spectral maths and does no file I/O. **They stay independent: neither
includes the other.** The only place they meet is the thin layer `../pipelines` (CMake target `nxr_pipelines`, linking `nxr_io` and `nxr_compute` directly): a *pipeline*
reads its inputs with nxr_io, computes with nxr_compute, writes the result with nxr_io, and **caches** — it looks for a stored result first and reuses it.

**Building it.** It is part of the one top-level build (`-DNXR_NEURO_BUILD_PIPELINES=OFF` leaves it out). Eigen 3.4.0 is fetched into the build directory (`../cmake/eigen.cmake`; an installed
Eigen 5 does not satisfy geometry-central's version check, and geometry-central would otherwise download one into its own source tree); `-DNXR_EIGEN_DIR=<an Eigen 3 source folder>` uses one you have.
The nxr_compute version recorded in results is the toolkit's version + this repository's commit (`0.1.0+26f6347eca`, `-dirty` if `compute/src`, `compute/include` or its `CMakeLists.txt` differ from it), read at
configure time: **re-run cmake after changing compute**. With the layer built, `nxr-neuro` has the `eigen` noun.

**The cache rule** (`nxr/pipelines/cache.h`, written once; every pipeline uses it). A result is found **by its rows, never by guessing paths**. The `producer_json` of the
pipeline's main product (the stored node) carries the identity of the job that made it, under `"cache"`:

    "cache": { "pipeline": "eigen_solve",                          the pipeline's name
               "params":   { "modes": 3000 },                      its parameters (the choices)
               "input":    { "mesh": "642:1b3c…", "positions": "…" },  the input's identity: nxr's mesh key `<nF>:<fnv1a>` (`nxr.geom.glue.mesh_key`, the Studio's `meshKey`) and the
                                                                   positions' own hash (a pial and a white surface share their faces)
               "compute":  "0.1.0+26f6347eca" }                    nxr-compute's version and commit

    find_cached(db, subject_id, table, key[, path]) -> the complete node whose cache equals the key, or nothing
    cached<R>(subject, table, path, key, load, compute, store)     HIT: load(row) reads the stored arrays, nothing is computed
                                                                   MISS: compute(), then store(result) inside sub.compose — rows and arrays together, all or nothing

A change of mesh (faces or positions), of a parameter, or of nxr-compute is a different key: a miss. Where a pipeline's product has a fixed name in nxr's layout (as the eigen
package's), a miss on a subject that already holds a package made by another job is **refused unless `--replace`**, as `nxr eigen solve` does; a replaced package's arrays are moved
aside to `<datastore>/_replaced-<date>/…`, never deleted.

**The first pipeline: `eigen_solve`** — `nxr-neuro eigen solve --dataset D/N --subject S [--modes K] [--surface NAME] [--replace]`, the twin of `nxr eigen solve` (same arguments, same JSON
lines; the result line adds `"cache": "hit" | "miss"`). It reads the stored surface, runs the Hodge–Dirac solve (nxr_compute: the intrinsic Delaunay triangulation, the DEC operators of that
triangulation, `hodgeLadders` = the reference method of nxr and Rheome by default, one shift-invert solve per ladder at σ = −1e-3, with `--shift/--slice/--normalize/--star1-floor` as named options: see the top-level README; the split solve of `nxr.eigen.hodge_dirac` — the Laplace–Beltrami and the 2-form ladders, one λ cut) and stores the package exactly as `nxr.io.eigen.store_package`
does (`eigenvalue/hodgeDirac`, `_kind`, `hodgeDirac_eigenvectors` [V × K] float32 sharded 64 modes, `hodgeDirac_support`). It was chosen because nxr-compute already has the Delaunay
intrinsic triangulation, the DEC operators on it and the eigensolver, so the job is complete **for a surface of one closed piece** (a mesh, a sphere; nxr solves those unglued too).
Check against nxr's own solve on the same icosphere: `test/parity/test_pipelines_parity.py` (eigenvalues to 1e-11, eigenvectors per degenerate subspace to 1e-7, rows, names, paths, dtypes,
chunks and codecs exact), with the cache timed (642 vertices: 36 ms to solve and store, ~1 ms to hit; 10 242 vertices, 300 modes: 3.3 s against 0.03 s of wall time with the process start).

    export NXR=<path to nxr> NXR_NEURO=$PWD/../build/cli/nxr-neuro
    uv run --project $NXR python -m pytest -q --basetemp=<a scratch folder> test/parity/test_pipelines_parity.py

**What `eigen_solve` still needs from nxr-compute** (nothing was ported into it):
1. **The glue by anatomy** (`nxr.geom.glue.glue_mesh` / the Studio's `glue`): the medial wall cut and a bridge grown between the rims, to make a cortex of two hemispheres one closed surface. Without it a
   surface of two pieces is refused (`NotImplementedError`, naming this). It also returns the glued surface and its bridge, which the package stores.
2. *(done, 2026-11-10)* Meyer's mixed Voronoi mass is nxr-compute's default `MassMatrixVariant::MixedVoronoi`, recorded on the mollified input before the flips by THE one Delaunay construction
   (`compute/include/nxr/delaunay.h`); the pipeline uses `ops::hodgedirac::assemble`, which is that construction, with the options `--mollify` and `--flip-eps` (recorded, in the cache key).
3. *(done, 2026-11-10)* The DEC accessors `d0(m)`, `d1(m)`, `hodge0..2(m)` follow the mesh option (the intrinsic Delaunay mesh by default); the intrinsic faces are `HodgeDirac::faces`.
4. For large meshes, streaming (`eigenvectors` is held whole, [V × K] float32 plus the solver's float64 modes) — nxr writes 64 modes at a time.

## Schema 53 and the measurement query (2026-11-17)

- **One table for values over members.** `selection_measurement` holds every value over a selection's members (a scalar row, or an array-backed row `[rows x members x components]` for a value that is not a
  scalar). The `statistic` table holds only values on axes that are not a selection's members; the task-level writer refuses a statistic with a member axis. `migrate_measurement_storage` (`migrate.h`) moves
  an older dataset's per-member statistics into measurements; it copies, verifies, and removes nothing unless asked.
- **Schema 53.** `selection_factor.level` (a product's factor may be a level of a stored grid), two query views (`member_value`, `product_factor_level`), `user_version` 53; a database at 50 to 52 is migrated in
  place when opened to write and read as it is when opened to read. A line's time grid is one `ladder` row, its rule closed form (`selection::save_time_grid`).
- **The query API** (`measurement_query.h`): filters (selection, member, level in physical units, subject or group, field, band, value) and roll-ups (a grid's closed-form hierarchy, one factor of a product,
  the whole, across subjects) in SQL for scalar rows, and by the same rule in memory (`rollup.h`) for arrays. Each measure's merge rule (sum, max, min, a mean through its weights, never a median) is data in
  `schema/measures.json` (`merge`), the one definition io and compute share; a derived measure is computed from its merged ingredients.

## Store format 51 (2026-11-14): standard Zarr, fewer files

- **Writers use the chunk key separator `.`:** `c.0.1` in the array's own folder (no `c/0/` subfolders; a 100 MB array went from 49 files in 49 folders to 3 files in none).
  A scalar's key stays `c`. `ArrayMeta::separator` carries it; `layout_array` removes the chunks of either encoding; a region, a slab and a populate follow the separator the
  array's own `zarr.json` declares.
- **Readers read both**, from each array's `zarr.json` (`ArrayReader`, `ZarrStore`, the writer's own plan). A dataset written before keeps its `c/0/1` arrays and stays readable and
  writable; its new arrays are `.`.
- **Large arrays are sharded by default** (see the array policy above); a small one stays one chunk.
- **Version decision: additive, `user_version` stays 50.** Nothing in the database changes (the rows describe an array's layout, not its key encoding, which is the array's `zarr.json`).
  A dataset made by this build records `meta.store_format = 51`; a dataset without the key is format 50 (an older reader reads its old arrays and refuses a `.` array with a clear message).
- Not converted: existing datasets (a later `dataset compact`). The `store_identity` golden is the new layout; `golden/store_identity_v50.txt` is kept and `store_identity_v50_layout`
  proves the old layout (`c/0/1`) is what the written tree becomes by renaming, byte for byte.
- **Every implementation follows (2026-11-14):** nxr (Python) writes `.`, applies the same default sharding policy (`array_meta`: same thresholds, inner chunk and shard shapes, index at the end, codecs)
  and records `meta.store_format = 51` (nxr #85); `io/test/parity/test_write_parity.py` and `test_session_parity.py` pass byte for byte. The Studio accepts the declared separator
  (`backend/src/db/import.ts`, `float16-write.ts`, `make-short-fixture.mjs`; Studio #232); `backend/src/read.ts` and `sharded.ts` already read it. `app/worker/index.ts:86` still preloads the
  demo's `c/0/0` keys: the demo store is built separately and keeps `/` until it is rebuilt.
