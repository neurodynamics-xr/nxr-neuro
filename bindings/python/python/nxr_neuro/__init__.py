"""nxr-neuro: measuring the spatiotemporal patterns of cortical activity from MEG/EEG.

The Python face of nxr-neuro: the pure functions over NumPy arrays (`nxr_neuro.compute`), the readers of the field's file formats (`nxr_neuro.io`) and the pipeline
level (`nxr_neuro.run`, `nxr_neuro.list`, `nxr_neuro.read`). Each calls the same C++ library as the `nxr-neuro` command; nothing here is a second implementation.

    import nxr_neuro as nn
    freqs, psd, *_ = nn.compute.signal.psd(x, sfreq=1000.0)
    r = nn.run("psd", "study/demo", "sub-01", recording="rest", all=True)   # a pipeline on a subject: the command line's `run`
"""

from importlib import metadata as _metadata

from . import compute, io
from . import list  # noqa: A004  (nxr_neuro.list: the lists of `nxr-neuro list`)
from ._pipeline import RunResult, Stored, read, run
from ._errors import (
    CancelledError,
    Error,
    FormatError,
    InvalidArgumentError,
    IOError_,
    NotConvergedError,
    UnsupportedError,
)

def __getattr__(name):
    if name == "ops":   # deprecated alias of `compute`, for one release: warns when used
        import warnings

        warnings.warn("nxr_neuro.ops is deprecated: use nxr_neuro.compute (the same functions)", DeprecationWarning, stacklevel=2)
        return compute
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    __version__ = _metadata.version("nxr-neuro")
except _metadata.PackageNotFoundError:  # a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "io",
    "compute",
    "list",
    "run",
    "read",
    "RunResult",
    "Stored",
    "Error",
    "FormatError",
    "UnsupportedError",
    "InvalidArgumentError",
    "IOError_",
    "NotConvergedError",
    "CancelledError",
    "__version__",
]
