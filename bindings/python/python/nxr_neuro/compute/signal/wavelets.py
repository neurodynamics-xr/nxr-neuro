"""The wavelet transforms of a signal in time (`nxr/signal/wavelet.h`, `nxr/signal/timefilterbank.h`): the analytic signal and envelope of bands, the time filterbank
for inspection, and `wt` / `iwt` of a bank.

A time bank is a dict of the library's design fields: `family` ("itersine", "meyer", "morlet", "morse", "amor", "bump", "power_partition"), `sides` ("analytic" for a
real signal, "two_sided" for a complex one; default "analytic"), `gain` (the peak in-band response, default 1) and the family's fields (`centres`, `cycles`, `bands`,
`edges`, `half_width`, `voices`, `anchor`, `limits`, `min_bins`, `gamma`, `time_bandwidth`, `edge_bins`). The bank is DESIGNED on the number of samples of the signal given:
that window is the caller's, and nothing is cut or padded beyond what the design says. Signals are [channels, samples]; coefficients [members, channels, samples], complex.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from ... import _core
from . import _banks


class TimeBank(NamedTuple):
    frequencies: np.ndarray            # Hz, the grid's bins (rfftfreq for sides="analytic", fftfreq for "two_sided")
    gains: np.ndarray                  # [members, bins], with the DC / Nyquist rule applied
    centre_frequencies: np.ndarray     # Hz, one per member
    frame_bounds: tuple                # (A, B): the minimum and maximum of the frame operator sum g^2 on the interior bins


def _bands(bands) -> np.ndarray:
    b = np.ascontiguousarray(bands, dtype=np.float64)
    if b.ndim == 1 and b.size == 2:
        b = b.reshape(1, 2)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("bands is [bands, 2]: the low and high edge of each, Hz")
    return b


def bandAnalytic(x, sfreq: float, bands, shape: str = "morlet", half_width: float = 1 / 3):
    """The analytic signal of bands (`nxr::signal::bandAnalytic`): complex [bands, channels, samples].

    One member per band [lo, hi] Hz of `shape` ("morlet": a Gaussian of full width at half maximum hi - lo; "meyer": Meyer edges, `half_width` octaves;
    "power_partition": squared gains adding to 1), gain 2 (the one-sided analytic signal), DC and Nyquist dropped, designed on x's own length: x is the window
    the caller hands in (periodic on it; the caller's halo holds the edge effect).

    x : array [channels, samples], real. bands : [bands, 2] (low, high) in Hz.
    """
    return _core.signal_bandAnalytic(x, float(sfreq), _bands(bands), shape, float(half_width))


def bandEnvelope(x, sfreq: float, bands, shape: str = "morlet", half_width: float = 1 / 3):
    """The envelopes of bands, |bandAnalytic| (`nxr::signal::bandEnvelope`): real [bands, channels, samples], in the input's precision. Arguments as `bandAnalytic`."""
    return _core.signal_bandEnvelope(x, float(sfreq), _bands(bands), shape, float(half_width))


def timeBank(
    sfreq: float,
    n: int,
    family: str,
    sides: str | None = None,
    gain: float | None = None,
    *,
    centres: Sequence[float] | None = None,
    cycles: float | None = None,
    bands=None,
    edges: Sequence[float] | None = None,
    half_width: float | None = None,
    voices: int | None = None,
    anchor: float | None = None,
    limits=None,
    min_bins: int | None = None,
    gamma: float | None = None,
    time_bandwidth: float | None = None,
    edge_bins: str | None = None,
) -> TimeBank:
    """A time filterbank designed on n samples at sfreq, for inspection (`nxr::signal::timefilterbank::design`): the grid, the gains and the frame bounds.

    `family`, `sides` and `gain` are required (the library's design has no default for them; `None` here only lets the error name the valid values): `sides` is "analytic" (a real
    signal, bins f >= 0) or "two_sided" (a complex signal, signed f), `gain` a number > 0. Every other field not given keeps the library's default. `edge_bins` ("single" or "drop") says what DC and Nyquist get; `gain` is the peak in-band response of each member.
    """
    if sides is None or sides not in ("analytic", "two_sided"):
        raise ValueError(f"timeBank: sides is required, one of 'analytic' or 'two_sided' (got {sides!r})")
    if gain is None or not float(gain) > 0:
        raise ValueError(f"timeBank: gain is required, the peak in-band response of each member, a number > 0 (got {gain!r})")
    bank = dict(family=family, sides=sides, gain=gain, centres=centres, cycles=cycles, bands=bands, edges=edges, half_width=half_width, voices=voices, anchor=anchor,
                limits=limits, min_bins=min_bins, gamma=gamma, time_bandwidth=time_bandwidth, edge_bins=edge_bins)
    f, g, c, a, b = _core.signal_timeBank(float(sfreq), int(n), _banks.plain(bank))
    return TimeBank(f, g, c, (a, b))


def wt(x, sfreq: float, bank) -> np.ndarray:
    """The wavelet transform of x [channels, samples] by a time bank (`nxr::signal::wt`): complex [members, channels, samples], at the full rate.

    `bank`: a dict as described in this module. A real signal for an analytic bank, a complex (or real) one for a two-sided bank. The bank is designed on x's number of samples.
    """
    return _core.signal_wt(_banks.field(x, "x"), float(sfreq), _banks.plain(bank))


def iwt(W, sfreq: float, n: int, bank, synthesis: str = "dual") -> np.ndarray:
    """The record back from coefficients W [members, channels, n] (`nxr::signal::iwt`, `iwtComplex`): real [channels, n] for an analytic bank, complex for a two-sided one.

    `bank` and `sfreq` are those of `wt`, and n the number of samples it was designed on. synthesis: "dual" (g / S, exact wherever the bank covers) or "tight" (g, exact when the bank is tight).
    """
    return _core.signal_iwt(_banks.stack(W, "W"), float(sfreq), int(n), _banks.plain(bank), synthesis)
