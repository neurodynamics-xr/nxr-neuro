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
| `tsv.h` | BIDS-style TSV tables (`n/a`, no empty cells, LF) and JSON sidecars: read and write |
| `recording.h` | `RecordingHeader` (channels, units, sampling rate, sample count, annotations, bad channels, gaps) and `RecordingReader::read(s0, n, c0, c1, out)`: block reads by sample range, in SI units, row after row, precision following the file |

## Rules every format follows

- Native: no MATLAB or Python at run time. MNE-Python and nibabel only generate the synthetic goldens (`test/golden/gen_*.py`, each next to its golden).
- A recording reader holds the file's directory and reads only what a call asks for; the result of a block read never depends on the block size.
- SI units on import; the file's own unit and scale stay in the channel.
- A reader parses every header field into the object. What an importer keeps of identifying fields is the importer's decision, not the reader's.
- An unsupported variant is refused with the reason, never guessed.
