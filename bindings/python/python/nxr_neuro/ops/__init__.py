"""The pure functions: arrays in, arrays out. No file, no dataset, no recording: a function never knows where its input came from.

Signals are NumPy arrays of shape [channels, samples]. A float32 or float64 input is computed in float64 by the library and the result comes back in the
input's precision (float64 results are handed over without a copy and may be Fortran-ordered).
"""

from . import filter, manifold, pca, welch

__all__ = ["filter", "manifold", "pca", "welch"]
