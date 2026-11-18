"""The joint time-vertex wavelet transform (`nxr/signal/jointfilterbank.h`, `nxr/signal/jointwavelet.h`): `jwt` and `ijwt`.

A member is the product of a graph-bank member, a time-bank member and (optionally) a joint kernel K(lambda, f): the members run over all combinations, the graph
index fastest. `graph_bank` and `time_bank` are the dicts of `graph.py` and `wavelets.py`. The time bank is designed on the field's own number of samples, at `sfreq`.
Kernels: a list of dicts, {"type": "speed", "c": ..., "d": ...} (a ridge on omega = c sqrt(lambda), positive f; c in length/s, d the angular width in rad/s) or
{"type": "ridge", "v": ..., "width": ..., "sense": +1 or -1} (a ridge on f = sense v sqrt(lambda) / 2 pi, `width` in Hz; a two-sided time bank).

The domain is given by the transform arguments: with only `eigenvalues`, F is a [modes, samples] series of mode coefficients; with `eigenvectors` too (or a `laplacian`
for the Chebyshev series, separable banks only), F is a field [vertices, samples]. A real series needs an analytic time bank, a complex one a two-sided bank.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ... import _core
from . import _banks


class JointCoefficients(NamedTuple):
    coefficients: np.ndarray     # complex [members, rows, samples]
    index: np.ndarray            # int64 [members, 3]: the graph member g, the time member t and the kernel k of each (0-based; g varies fastest)


def jwt(F, sfreq: float, graph_bank, time_bank, *, eigenvalues=None, eigenvectors=None, mass=None, laplacian=None, order: int = 40, lmax: float = 0.0,
        kernels=None) -> JointCoefficients:
    """The joint wavelet transform of F [rows, samples] (`nxr::signal::jwt`): complex [members, rows, samples] and the member index table."""
    ev, evec, m, lap = _banks.transform(eigenvalues, eigenvectors, mass, laplacian)
    w, index = _core.signal_jwt(_banks.field(F, "F"), float(sfreq), _banks.plain(graph_bank), _banks.plain(time_bank), ev, evec, m, lap, int(order), float(lmax),
                                [_banks.plain(k) for k in (kernels or [])])
    return JointCoefficients(w, index)


def ijwt(W, sfreq: float, n: int, graph_bank, time_bank, *, eigenvalues=None, eigenvectors=None, mass=None, laplacian=None, order: int = 40, lmax: float = 0.0,
         kernels=None, synthesis: str = "dual") -> np.ndarray:
    """The series back from coefficients W [members, rows, n] (`nxr::signal::ijwt`, `ijwtComplex`): [rows, n], real for an analytic time bank, complex for a two-sided one.

    Arguments as `jwt`; n the number of samples the time bank was designed on; synthesis "dual" (exact wherever the bank covers) or "tight".
    """
    ev, evec, m, lap = _banks.transform(eigenvalues, eigenvectors, mass, laplacian)
    return _core.signal_ijwt(_banks.stack(W, "W"), float(sfreq), int(n), _banks.plain(graph_bank), _banks.plain(time_bank), ev, evec, m, lap, int(order), float(lmax),
                             [_banks.plain(k) for k in (kernels or [])], synthesis)
