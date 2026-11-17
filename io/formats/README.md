# io/formats: pure readers and writers

The field's file formats as plain C++: bytes in, an in-memory object out, and back. Design: `docs/specs/2026-11-17-interoperability.md`, section 3.1.
They build into their own target, `nxr_io_formats`, which `nxr_io` links; a wheel, a MEX or a WebAssembly client can use a format without the store.

## The rule

A pure format touches no store, database, writer, dataset, subject or package, and no chunk service. The one `nxr/io` header it includes is
`errors.h`. `cmake/check_formats_layering.cmake` enforces it (the tests `formats_layering` and `formats_layering_check_catches`).
Calibration and unit scaling are decoding and belong here; any other maths does not.

## What is here

| Header (`include/nxr/formats/`) | What |
|---|---|
| `errors.h` | failures as `nxr::io::Error` with a code: `format_error` (bad bytes), `unsupported` (a valid file the reader refuses, with the reason), `invalid_argument` (a range outside the recording), `io` |
| `byte_source.h` | `ByteSource` (`size()`, `read(offset, n)`) with file and memory implementations; `ByteSink` for writers |
| `endian.h` | loads and stores in a stated byte order |
| `deflate.h` | zlib and gzip streams as byte vectors (zlib) |
| `nifti.h` | NIfTI-1/2: header (every field, extensions), volume (whole, `.nii`, `.nii.gz`, `.hdr`+`.img`), the best affine (sform, else qform, else pixdims), writer |
| `gifti.h` | GIFTI: surfaces, shape and functional maps, labels; ASCII, Base64Binary and GZipBase64Binary in either byte order; column-major made row-major; reader and writer (pugixml) |
| `cifti.h` | CIFTI-2 (dtseries, dscalar, dlabel, pscalar, ptseries, any two-dimensional map): the NIfTI-2 header and the XML extension of code 32 parsed whole (series, named maps and label tables, brain models with vertices and voxels, parcels, surfaces, the volume), the matrix by rectangles of rows and columns; read only |
| `tsv.h` | BIDS-style TSV tables (`n/a`, no empty cells, LF) and JSON sidecars: read and write |
| `edf.h` | EDF, EDF+, BDF, BDF+ (continuous and discontinuous: gaps as spans, read as zeros); annotations; several sampling rates refused unless one is named |
| `brainvision.h` | BrainVision `.vhdr`/`.vmrk`/`.eeg`: INT_16, INT_32, IEEE_FLOAT_32, multiplexed or vectorized; ASCII refused; markers as annotations; coordinates |
| `eeglab.h` | EEGLAB `.set` (+ `.fdt`) through matio: events, channel locations, the `.fdt` by seek, data in an uncompressed `.set` by block; compressed variables and epoched datasets refused |
| `fif.h` | FIF (Neuromag/MNE): the tag layer (`File`: tag directory, block tree, tag values; read) and the raw MEG/EEG reader (`RawReader`): every `meas_info` field (channels with `cal`/`range`/`loc`, `subject_info`, `meas_date`, device and head transforms, `dig`, projectors, compensation, bads, HPI, helium, events, annotations, unmodelled blocks as generic tags), blocks by sample range, int16/int32/float32/float64/complex, skips as gaps, split files as one recording; epoched, evoked and non-raw FIF refused with the reason; ported from MNE-Python (BSD-3-Clause) |
| `fif_write.h` | FIF tag writing (`TagWriter`): file id, blocks, ints, floats and doubles, strings, name lists, ids, channel info, coordinate transforms, dense and sparse matrices, generic tags; as MNE-Python writes them |
| `fif_objects.h` | FIF objects: source spaces (`-src.fif`: surface, volume, discrete), noise covariances (`-cov.fif`), forward solutions (`-fwd.fif`) and inverse operators (`-inv.fif`) (read), event lists (`-eve.fif`), annotation files (`-annot.fif`), transforms (`-trans.fif`); write for all but the forward and inverse; every tag parsed, the rest kept as generic tags |
| `bids.h` | BIDS: entities and file names (`parse_name`, `build_name`, the specification's entity order), a folder `Source` (directory or in memory), the dataset parsed into subjects, sessions and recordings (merged sidecars by the inheritance principle, channels, events, electrodes, coordinate systems, scans), `dataset_description.json`, `participants.tsv`/`.json`; pure, no import |
| `ctf.h` | CTF `.ds` folders: the `.res4` header (every field, channels with coils, filters, compensation), `.meg4` trials and continuation files by block (seeks per channel and trial), the system-clock truncation, `.hc` head coils and the transforms derived from them, markers, `bad.segments`, `BadChannels`, `.eeg` and `.pos`; epoched and averaged data sets refused; ported from MNE-Python |
| `kit.h` | KIT/Yokogawa `.con`/`.sqd`: the directory, system, channel, calibration, amplifier, acquisition, coregistration and digitization blocks (every field), int16 or int32 samples by block; evoked, epoched and pre-V2R003 files refused; ported from MNE-Python |
| `units.h` | file unit strings to SI (`uV`, `µV`, `mV`, `fT`, ...) |
| `recording.h` | `RecordingHeader` (channels, units, sampling rate, sample count, annotations, bad channels, gaps) and `RecordingReader::read(s0, n, c0, c1, out)`: block reads by sample range, in SI units, row after row, precision following the file (a `double*` overload for 64-bit recordings) |

## Rules every format follows

- Native: no MATLAB or Python at run time. MNE-Python and nibabel generate the synthetic goldens (`test/golden/gen_*.py`, each next to its golden); the one exception to "no code of a reference" is FIF, ported from MNE-Python with attribution (`THIRD_PARTY_NOTICES.md`).
- A recording reader holds the file's directory and reads only what a call asks for; the result of a block read never depends on the block size.
- SI units on import; the file's own unit and scale stay in the channel.
- A reader parses every header field into the object. What an importer keeps of identifying fields is the importer's decision, not the reader's.
- An unsupported variant is refused with the reason, never guessed.

## FIF details

- **Channel kinds** (`Channel::kind`, from `FIFF_CH_KIND`): `meg`, `meg_ref`, `eeg`, `stim`, `eog`, `emg`, `ecg`, `misc`, `resp`, `seeg`, `dbs`, `ecog`, `fnirs`, `bio`, `mcg`, `syst`, `ias`, `exci`, `eyetrack`, `gsr`, `temperature`, `dipole_wave`, `goodness_fit` (`fif::channel_kind_name`). The coil type, which tells magnetometers from gradiometers, is MNE's name in `Channel::coil_type` (`VV_MAG_T3`, `VV_PLANAR_T1`, `CTF_GRAD`, ...), without the compensation grade held in its high 16 bits (`info().compensation_grade`).
- **Values:** `stored x cal x range`, in the unit of `FIFF_CH_UNIT` (T, T/m, V, ...). MNE does not apply `FIFF_CH_UNIT_MUL` and neither does the reader (it is in `info().channels`). Active projectors are not applied (`info().projectors` has them, with `active`).
- **Time:** sample 0 is the first buffer; `info().first_samp` is the file's first sample number, so `start_time` is `meas_date + first_samp / sfreq`. A skip before the first buffer moves `first_samp`; between buffers it is a gap read as zeros.
- **Header scope:** the tags and blocks of the measurement that have no field (Maxfilter history, cross-talk and fine-calibration data, vendor blocks) are in `info().other_tags` / `other_blocks` as generic tags, so nothing is dropped.
- **Not read yet:** epochs/evoked, `.stc`, forward-solution and inverse-operator writers (spec section 7, steps 8 and 11). `data/mne/coil_def.dat` (MNE's coil definitions, BSD-3-Clause, with its licence) is vendored for the forward model and not read by any code yet.

## FIF objects

- **Files:** read through `fif::File` (the tag layer); `read_source_spaces`, `read_covariance(s)`, `read_forward`, `read_inverse`, `read_events`, `read_annotations_file`, `read_transform`. Each refuses a file that holds none of its kind, with the reason.
- **As stored:** SI units (metres, tesla), the file's 32-bit floats widened to double, matrices row-major as the file stores them. The named matrices of a forward solution and an inverse operator are NOT transposed (MNE transposes them on reading; the tests compare against the transpose). Vertex and triangle indices are zero-based (the file's are one-based).
- **Covariance:** the file's packed lower triangle is given as the full symmetric matrix; a diagonal covariance as its diagonal. A sparse covariance is refused.
- **Writers** follow MNE's tag order, so a file read and written again is byte-identical to the MNE-written file (the tests check it). JSON inside tags (annotation channel names and extras) is written as Python's `json.dumps` writes it.
- **Not modelled:** blocks of a file that no field takes are dropped; tags of a modelled block that no field takes are kept in `other_tags` and written back.

## BIDS details

- **Entities** follow the specification's order (`sub`, `ses`, `sample`, `task`, `tissue`, `acq`, `ce`, `trc`, `stain`, `rec`, `dir`, `run`, `mod`, `echo`, `flip`, `inv`, `mt`, `part`, `proc`, `hemi`, `space`, `split`, `recording`, `chunk`, `atlas`, `res`, `den`, `label`, `desc`); `build_name` writes in that order (MNE-BIDS orders a few entities differently; the goldens compare only what both order alike). A name that is not `key-value` entities, a suffix and an extension is refused by `parse_name`; the dataset reader lists such a file in `issues` (or refuses with `ReadOptions::strict`), it never drops it silently.
- **Inheritance:** a metadata file applies to a data file when its suffix is named and its entities are all in the data file's name (for electrodes and coordinate systems, `space` need not be); JSON files are merged from the root down (a deeper value replaces a shallower one whole), tables (channels, events) take the most specific file, electrodes and coordinate systems one file per `space`.
- **Recordings:** data files of the `meg`, `eeg` and `ieeg` datatypes; split parts are one recording (the `split` entity is dropped from its entities, parts in numeric order); BrainVision `.vmrk`/`.eeg` and EEGLAB `.fdt` are companions; a CTF `.ds` folder is one data file. The data are never read here.
## CTF and KIT details

- **CTF values:** `stored x 1 / (proper_gain x qgain)`, in T for MEG and reference channels and V for the others; a MEG channel whose coil has no position is a `misc` channel (MNE's rule). `Channel::location` and `orientation` are the first coil's position (m) and normal in the CTF device frame, as the file has them; the head coils and the transforms MNE derives from them are `coils()` and `transforms()`.
- **CTF time:** the trials of the `.meg4` file and of its continuation files (`.1_meg4`, ...) are one recording, in order, as MNE reads them. The recording stops at the first zero of the system clock channel unless `OpenOptions::system_clock` is `ignore`. A data set with pre-trigger points or several averages is refused as not continuous.
- **KIT values:** MEG channels `stored x adc_range / 2^adc_stored / amp_gain x gain` (T); EEG and ECG channels with their own gain; the others `stored x adc_range / 2^adc_stored` (V). No stimulus channel is synthesised from the trigger channels. The digitization in the file is given as stored (digitizer frame, metres); the transforms are the importer's.
- **Goldens:** MNE-Python cannot write CTF or KIT files, so `test/golden/gen_ctf.py` and `gen_kit.py` write minimal valid files from the format descriptions (layouts as MNE-Python's readers parse them); MNE-Python reading them is the check that they are valid and the source of every golden.

## CIFTI-2 details

- **Orientation** is nibabel's: the matrix is `rows() x cols()`, rows indexing matrix dimension 0 (the series or the named maps) and columns dimension 1 (the grayordinates or parcels); in the file dimension 0 varies fastest, so `read` makes one contiguous read per column and a time series of a few grayordinates is cheap.
- **Types:** float32 and float64 files; other data types, CIFTI-1 files (version other than 2 or 2.0), a NIfTI-1 file, a file with no extension of code 32 and gzipped files are refused with the reason. A mapping that does not cover the data's dimension is refused.
- **Not here yet:** the writer (spec section 5.1, after GIFTI export) and the surface templates a CIFTI file's vertex counts are matched against (the importer's job).
