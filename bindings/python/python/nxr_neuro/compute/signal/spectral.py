"""Welch's power spectral density and cross-spectral density (`nxr/signal/spectral.h`).

A record x[0..T) of C channels is cut into segments of `segment_samples` (n) samples, a new segment every hop = n - floor(n * overlap) samples,
from sample 0, as many as fit whole (a partial last segment is dropped, as scipy.signal.welch and Brainstorm do). Each segment is detrended,
multiplied by a window w and transformed (rfft), and |X_k|^2 is averaged over the segments. The one-sided spectrum doubles every bin but DC (and,
for an even n, Nyquist). Frequencies are k * sfreq / n, k = 0 .. n/2.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from ... import _core

_SCALING = """\
    scaling : {"density", "spectrum", "chain"}
        "density": |X_k|^2 / (sfreq * sum w^2), power per Hz (scipy scaling="density"; Brainstorm's "physical" units; the default).
        "spectrum": |X_k|^2 / (sum w)^2, power (scipy scaling="spectrum").
        "chain": |X_k|^2 / (n * sum w^2), power per sample, per bin (the Studio's and nxr's chain Welch)."""


class WelchResult(NamedTuple):
    frequencies: np.ndarray        # Hz, segment_samples // 2 + 1 of them
    psd: np.ndarray                # [channels, frequencies], in the input's precision
    segments: int                  # the segments averaged (0: the record held none; psd is then zero)
    segments_skipped: int          # whole segments left out for touching a bad sample


class CrossSpectrumResult(NamedTuple):
    frequencies: np.ndarray        # Hz, of all bins
    real: np.ndarray               # [groups, channels, channels]: the real part of the cross-spectral density of each group of bins
    imag: np.ndarray | None        # [groups, channels, channels], antisymmetric: its imaginary part (the quadrature spectrum); None unless asked
    segments: int
    segments_skipped: int


def psd(
    x,
    sfreq: float,
    segment_samples: int = 256,
    overlap: float = 0.5,
    window: str = "hann",
    periodic: bool = True,
    detrend: str = "constant",
    scaling: str = "density",
    good=None,
) -> WelchResult:
    """Welch's power spectral density of every channel (`nxr::signal::spectral::Welch`).

    Parameters
    ----------
    x : array [channels, samples]
        The record, float32 or float64. The result comes back in the same precision.
    sfreq : float
        Sampling frequency, Hz. It sets the frequencies and the density.
    segment_samples : int
        n, samples per segment (default 256).
    overlap : float
        Fraction of a segment shared with the next, in [0, 1): hop = n - floor(n * overlap) (default 0.5).
    window : {"hann", "hamming", "blackman", "rectangular"}
        The window (default "hann").
    periodic : bool
        True: the DFT-even form of the window (w_j = f(2 pi j / n), scipy and numpy-based nxr; default). False: the symmetric form
        (w_j = f(2 pi j / (n - 1)), MATLAB's hamming(n)).
    detrend : {"none", "constant", "linear"}
        Removed from each segment before the window (default "constant", scipy's).
""" + _SCALING + """
    good : array of bool [samples], optional
        A segment that touches a sample that is not good is left out (Brainstorm's bad segments). The segment grid does not move.

    Returns
    -------
    WelchResult
        (frequencies, psd, segments, segments_skipped). The values are those of scipy.signal.welch for the same segmentation, window, detrend and
        scaling (see `scipyWelchOptions`).
    """
    return WelchResult(*_core.signal_psd(x, float(sfreq), int(segment_samples), float(overlap), window, bool(periodic), detrend, scaling, good))


def xpsd(
    x,
    sfreq: float,
    bins: Sequence[tuple[int, int]],
    segment_samples: int = 256,
    overlap: float = 0.5,
    window: str = "hann",
    periodic: bool = True,
    detrend: str = "constant",
    scaling: str = "density",
    mean: bool = True,
    quadrature: bool = False,
    good=None,
) -> CrossSpectrumResult:
    """Welch's cross-spectral density between every pair of channels, accumulated per group of bins (`nxr::signal::spectral::CrossWelch`).

    S_g = scale / segments x (mean or sum over the group's bins of w_k Re(X_k X_k^H)), w_k the one-sided doubling. The same segments, detrending,
    window, scaling and bad-segment rule as `psd`, which is its diagonal. Memory: channels^2 per group, never a spectrum per pair.

    Parameters
    ----------
    x : array [channels, samples]
        The record, float32 or float64.
    sfreq : float
        Sampling frequency, Hz.
    bins : sequence of (first, last)
        The groups: each a run of bins first..last (inclusive) of the one-sided spectrum. Bin k is at k * sfreq / segment_samples Hz.
    segment_samples, overlap, window, periodic, detrend, scaling, good
        As in `psd`.
    mean : bool
        True: a group's value is the mean of its bins (a band; default). False: their sum (a total).
    quadrature : bool
        Also accumulate the imaginary part Im C (the quadrature spectrum), Im(X_k X_k^H) = B A^T - A B^T for X = A + iB. The complex density of a
        group is real + 1j * imag.

    Returns
    -------
    CrossSpectrumResult
        (frequencies, real, imag, segments, segments_skipped).
    """
    b = [(int(a), int(c)) for a, c in bins]
    return CrossSpectrumResult(
        *_core.signal_xpsd(
            x, float(sfreq), int(segment_samples), float(overlap), window, bool(periodic), detrend, scaling, b, bool(mean), bool(quadrature), good
        )
    )
