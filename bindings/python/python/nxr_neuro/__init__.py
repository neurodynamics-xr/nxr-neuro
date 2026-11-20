"""nxr-neuro: measuring the spatiotemporal patterns of cortical activity from MEG/EEG.

The Python face of nxr-neuro: the pure functions over NumPy arrays (`nxr_neuro.compute`), the readers of the field's file formats (`nxr_neuro.io`), and every
command of the `nxr-neuro` command line as a function: `nxr-neuro <verb> <noun> --option VALUE` is `nxr_neuro.<verb>.<noun>(option=VALUE)` (dashes as underscores; the
verb `import` is `nxr_neuro.import_`, the option `--from` is `from_`), `nxr-neuro run` is `nxr_neuro.run`. Each calls the same C++ library as the command line; nothing
here is a second implementation.

    import nxr_neuro as nn
    freqs, psd, *_ = nn.compute.signal.psd(x, sfreq=1000.0)
    nn.create.dataset(dataset="study/demo"); nn.create.subject(dataset="study/demo", name="sub-01")
    r = nn.run("psd", "study/demo", "sub-01", recording="rest", all=True)   # a pipeline on a subject: the command line's `run`
    psd = nn.read.statistic(dataset="study/demo", subject="sub-01", name="rest_psd_welch").arrays["values"]
"""

from importlib import metadata as _metadata

import sys as _sys

from . import compute, io
from . import _commands
from ._commands import Dataset, Result, open, run  # noqa: A004  (nxr_neuro.open: a handle on a dataset)
from ._errors import (
    Ambiguous,
    AlreadyExists,
    CancelledError,
    Error,
    FormatError,
    InvalidArgumentError,
    IOError_,
    NoSuchRow,
    NotADatabaseError,
    NotConvergedError,
    NotFound,
    NotWritable,
    SchemaVersionError,
    UnsupportedError,
    WriterLockHeld,
)

# the verbs: nxr_neuro.create, .list, .show, .read, .add, .import_, .export, ... (one function a noun) and nxr_neuro.compile
_commands.install(_sys.modules[__name__])

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
    *sorted({_commands.py_name(c[0]) for c in _commands.COMMANDS}),
    "run",
    "open",
    "Dataset",
    "Result",
    "Error",
    "FormatError",
    "UnsupportedError",
    "InvalidArgumentError",
    "IOError_",
    "NotConvergedError",
    "CancelledError",
    "WriterLockHeld",
    "NotWritable",
    "SchemaVersionError",
    "NotADatabaseError",
    "NoSuchRow",
    "Ambiguous",
    "AlreadyExists",
    "NotFound",
    "__version__",
]
