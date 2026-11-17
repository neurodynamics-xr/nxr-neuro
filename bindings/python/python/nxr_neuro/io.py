"""Readers of the field's file formats (`io/formats`): native C++, no MATLAB and no Python library at run time.

Every reader parses the header whole and gives it as a dictionary; signal values come as NumPy arrays in SI units (a file's microvolts come out as
volts; the file's own unit and scale stay in the channel). A recording is never loaded whole unless the caller says so: the time span is always the
caller's (`start` and `stop` in samples, or `all=True`).

    rec = nn.io.open_edf("rest.edf")
    rec.header["sampling_rate"], rec.header["n_samples"]
    x = rec.read(start=0, stop=10_000)          # [channels, samples]
    x, header = nn.io.read_edf("rest.edf", all=True)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _core

__all__ = ["Recording", "open_edf", "open_bdf", "open_brainvision", "open_eeglab", "read_edf", "read_bdf", "read_brainvision", "read_eeglab",
           "read_nifti", "read_gifti", "read_tsv"]


class Recording:
    """An open recording: the header now, samples by block (`nxr::formats::RecordingReader`)."""

    def __init__(self, core: _core.Recording):
        self._core = core
        self._header: dict[str, Any] | None = None

    @property
    def header(self) -> dict[str, Any]:
        """The parsed header: format, channels (name, kind, unit, file_unit, scale_to_si, gain, offset, location, bad, ...), sampling_rate (Hz),
        n_samples (per channel), precision, start_time, annotations (onset and duration in seconds, label), bad_channels, gaps (first sample,
        count), position_unit, and info (every other field, by the file's own name)."""
        if self._header is None:
            self._header = self._core.header()
        return self._header

    def read(self, start: int, stop: int, first_channel: int = 0, last_channel: int | None = None) -> np.ndarray:
        """Samples [start, stop) of channels [first_channel, last_channel) as [channels, samples], C-ordered, in SI units.

        float32 for a file that holds integers or 32-bit floats, float64 for a 64-bit recording (precision follows the file). Samples in a gap
        (`header["gaps"]`) read as zeros. The result never depends on the block read. A span outside the recording raises `InvalidArgumentError`.
        """
        return self._core.read(int(start), int(stop), int(first_channel), None if last_channel is None else int(last_channel))

    def read_all(self, first_channel: int = 0, last_channel: int | None = None) -> np.ndarray:
        """The whole recording (the span stated as everything): memory is the recording's size."""
        return self.read(0, self.header["n_samples"], first_channel, last_channel)


def open_edf(path: str, sampling_rate: float = 0.0) -> Recording:
    """Open an EDF, EDF+, BDF or BDF+ file (continuous or discontinuous; the format is recognised from the header).

    `sampling_rate` (Hz): the rate of the signals to read; 0 = the file's only rate (a file with several rates is refused, the refusal lists them).
    Annotation signals become `header["annotations"]` and are not channels; the discontinuous records' gaps are `header["gaps"]`.
    """
    return Recording(_core.open_edf(str(path), float(sampling_rate)))


open_bdf = open_edf


def open_brainvision(path: str) -> Recording:
    """Open a BrainVision recording by its `.vhdr` (with the `.vmrk` markers and the `.eeg` data): INT_16, INT_32 or IEEE_FLOAT_32, multiplexed or
    vectorized; an ASCII file is refused. Markers become `header["annotations"]`."""
    return Recording(_core.open_brainvision(str(path)))


def open_eeglab(path: str) -> Recording:
    """Open an EEGLAB `.set` (with its `.fdt`, or the data embedded in the `.set`): events and channel locations in the header; compressed variables
    and epoched datasets are refused."""
    return Recording(_core.open_eeglab(str(path)))


def _read(rec: Recording, start, stop, all, first_channel, last_channel):
    if all:
        if start is not None or stop is not None:
            raise ValueError("give start and stop, or all=True, not both")
        x = rec.read_all(first_channel, last_channel)
    else:
        if start is None or stop is None:
            raise ValueError("the span is the caller's: give start and stop (samples), or all=True")
        x = rec.read(start, stop, first_channel, last_channel)
    return x, rec.header


def read_edf(path: str, start: int | None = None, stop: int | None = None, all: bool = False, first_channel: int = 0,
             last_channel: int | None = None, sampling_rate: float = 0.0):
    """Read samples [start, stop) (or `all=True`) of an EDF/EDF+/BDF/BDF+ file: (data [channels, samples], header). See `open_edf`, `Recording.read`."""
    return _read(open_edf(path, sampling_rate), start, stop, all, first_channel, last_channel)


read_bdf = read_edf


def read_brainvision(path: str, start: int | None = None, stop: int | None = None, all: bool = False, first_channel: int = 0,
                     last_channel: int | None = None):
    """Read samples [start, stop) (or `all=True`) of a BrainVision recording: (data [channels, samples], header). See `open_brainvision`."""
    return _read(open_brainvision(path), start, stop, all, first_channel, last_channel)


def read_eeglab(path: str, start: int | None = None, stop: int | None = None, all: bool = False, first_channel: int = 0,
                last_channel: int | None = None):
    """Read samples [start, stop) (or `all=True`) of an EEGLAB `.set` recording: (data [channels, samples], header). See `open_eeglab`."""
    return _read(open_eeglab(path), start, stop, all, first_channel, last_channel)


def read_nifti(path: str):
    """Read a NIfTI-1 or NIfTI-2 volume (`.nii`, `.nii.gz`, or a `.hdr`/`.img` pair named by either): (data, header).

    `data` has the shape the file states, with the file's x axis first (the file stores x fastest: the array is Fortran-ordered), scaled by
    scl_slope and scl_inter when the file sets them. 8- and 16-bit integers and float32 read as float32; 32- and 64-bit integers and float64 as
    float64. Complex, RGB and 128-bit types are refused. `header` holds every field of the file's header; `header["affine"]` is the 4 x 4 voxel to
    world map of the best form (sform, else qform, else the pixel sizes), in `header["spatial_unit"]` (`unit_to_metres` converts).
    """
    return _core.read_nifti(str(path))


def read_gifti(path: str) -> dict[str, Any]:
    """Read a GIFTI file (surfaces, shape and functional maps, label maps): a dictionary with `version`, `metadata`, `labels` and `arrays`.

    Each array is a dictionary: `intent` ("NIFTI_INTENT_POINTSET", "NIFTI_INTENT_TRIANGLE", "NIFTI_INTENT_SHAPE", ...), `datatype`, `metadata`,
    `transforms` and `data`, a NumPy array of the file's dimensions in row-major order whatever the file's indexing order (float32, int32 or uint8).
    ASCII, Base64Binary and GZipBase64Binary encodings are read in either byte order; ExternalFileBinary is refused.
    """
    return _core.read_gifti(str(path))


def read_tsv(path: str) -> dict[str, list[str]]:
    """Read a BIDS-style TSV table: a dictionary from column name to the list of its cells, as text (`n/a` is kept as written; no cell is empty).

    A row whose cell count differs from the header's is refused with its line number.
    """
    return _core.read_tsv(str(path))
