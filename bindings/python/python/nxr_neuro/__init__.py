"""nxr-neuro: measuring the spatiotemporal patterns of cortical activity from MEG/EEG.

This first slice of the Python face holds the pure functions over NumPy arrays (`nxr_neuro.ops`) and the readers of the field's file formats
(`nxr_neuro.io`). Both call the same C++ library as the `nxr-neuro` command; nothing here is a second implementation.

    import nxr_neuro as nn
    freqs, psd, *_ = nn.ops.welch.psd(x, sfreq=1000.0)
"""

from importlib import metadata as _metadata

from . import io, ops
from ._errors import (
    CancelledError,
    Error,
    FormatError,
    InvalidArgumentError,
    IOError_,
    NotConvergedError,
    UnsupportedError,
)

try:
    __version__ = _metadata.version("nxr-neuro")
except _metadata.PackageNotFoundError:  # a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "io",
    "ops",
    "Error",
    "FormatError",
    "UnsupportedError",
    "InvalidArgumentError",
    "IOError_",
    "NotConvergedError",
    "CancelledError",
    "__version__",
]
