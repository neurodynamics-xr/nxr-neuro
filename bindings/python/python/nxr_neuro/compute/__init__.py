"""The pure functions: arrays in, arrays out. No file, no dataset, no recording: a function never knows where its input came from.

One word names a function in every layer: the toolbox (`operators`, `signal`, `stats`) and the C++ function (`eigen`, `psd`, `xpsd`, `lowpass`, `pca`).
Signals are NumPy arrays of shape [channels, samples]. A float32 or float64 input is computed in float64 by the library and the result comes back in the
input's precision (float64 results are handed over without a copy and may be Fortran-ordered).
"""

from . import operators, signal, stats

__all__ = ["operators", "signal", "stats"]

# the modules renamed to the toolbox words: an old name is refused with the new one (no alias)
_RENAMED = {
    "welch": "signal (psd, xpsd)",
    "filter": "signal (lowpass, highpass, bandpass, bandstop, notch)",
    "pca": "stats (pca, reconstruct)",
    "manifold": "operators (eigen)",
}


def __getattr__(name):
    if name in _RENAMED:
        raise AttributeError(f"nxr_neuro.compute.{name} was renamed: use nxr_neuro.compute.{_RENAMED[name]}")
    raise AttributeError(f"module 'nxr_neuro.compute' has no attribute {name!r}")
