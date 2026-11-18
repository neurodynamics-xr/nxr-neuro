"""The wavelets of a field on a graph (`nxr/signal/graphfilterbank.h`, `nxr/signal/graphwavelet.h`): the graph bank for inspection, and `gwt` / `igwt`.

A graph bank is a dict of the library's design fields: `family` ("mexhat", "heat", "itersine", "logitersine" designed on a spectral range; "meyer" on `edges`; "weyl" on
`area` and `levels`; "rank" on `levels`) and `lmax`, `lmin`, `num_filters`, `voices`, `lp_factor`, `fine_limit` ("basis" or "usable"), `scale_limits`, `half_width`. A bank
given no `lmax` is designed on the transform's own spectrum (the eigenvalues, or the series' bound).

A field is transformed through the operator's modes (`eigenvalues` and `eigenvectors` [vertices, modes], mass-orthonormal; `mass` a sparse matrix or its diagonal) or by a
Chebyshev series in the pencil (`laplacian` and `mass` sparse matrices, `order`, `lmax`: the series' bound, 0 to estimate it). Fields are [vertices, samples]; coefficients
[members, vertices, samples].
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from ... import _core
from . import _banks


class GraphBank(NamedTuple):
    gains: np.ndarray | None           # [members, K] at the eigenvalues given, else None
    roles: list                        # "member", "scaling", "wavelet" or "top", per member
    scales: np.ndarray                 # per member; NaN where the family has none
    centre_wavenumbers: np.ndarray     # per member; the gain-weighted centroid of sqrt(lambda) over the bank's range; NaN for a bank with no range (meyer, weyl, rank)
    frame_bounds: tuple                # (A, B) of sum g^2 on the eigenvalues given, else on the bank's own range; NaN for a bank with neither (meyer, weyl, rank)


def graph_bank(
    family: str,
    eigenvalues=None,
    lmax: float | None = None,
    lmin: float | None = None,
    num_filters: int = 0,
    voices: int = 3,
    lp_factor: float = 20,
    fine_limit: str = "basis",
    scale_limits=None,
    edges: Sequence[float] | None = None,
    area: float | None = None,
    levels: int | None = None,
    half_width: float = 1 / 3,
) -> GraphBank:
    """A graph filterbank, for inspection (`nxr::signal::graphfilterbank::make`): its members, and its gains and frame bounds at `eigenvalues` when given.

    The range is `lmin` and `lmax`, or `lmax` (lmin = lmax / lp_factor), or the eigenvalues themselves. `num_filters` 0 derives the wavelet count from `voices` per octave.
    `fine_limit`: "basis" (the finest scale tiles the whole spectrum) or "usable" (every member responds fully). "meyer" takes ascending `edges` in lambda, "weyl"
    the surface's `area` and `levels` (one octave a level), "rank" `levels`.
    """
    bank = dict(family=family, lmax=lmax, lmin=lmin, num_filters=num_filters, voices=voices, lp_factor=lp_factor, fine_limit=fine_limit, scale_limits=scale_limits,
                edges=edges, area=area, levels=levels, half_width=half_width)
    g, roles, scales, centres, a, b = _core.signal_graph_bank(_banks.plain(bank), _banks.vector(eigenvalues))
    return GraphBank(g, roles, scales, centres, (a, b))


def gwt(F, bank, *, eigenvalues=None, eigenvectors=None, mass=None, laplacian=None, order: int = 40, lmax: float = 0.0) -> np.ndarray:
    """The wavelet transform of a field F [vertices, samples] on a graph (`nxr::signal::gwt`): [members, vertices, samples].

    `bank`: a graph bank dict. Give the operator's modes (`eigenvalues`, `eigenvectors`, optionally `mass`: exact) or a `laplacian` (with its `mass`; `order` and `lmax`: the series).
    """
    ev, evec, m, lap = _banks.transform(eigenvalues, eigenvectors, mass, laplacian)
    return _core.signal_gwt(_banks.matrix(F), _banks.plain(bank), ev, evec, m, lap, int(order), float(lmax))


def igwt(W, bank, *, eigenvalues=None, eigenvectors=None, mass=None, laplacian=None, order: int = 40, lmax: float = 0.0, synthesis: str = "dual") -> np.ndarray:
    """The field back from coefficients W [members, vertices, samples] (`nxr::signal::igwt`): [vertices, samples]. Arguments as `gwt`; synthesis "dual" or "tight"."""
    ev, evec, m, lap = _banks.transform(eigenvalues, eigenvectors, mass, laplacian)
    return _core.signal_igwt(_banks.stack(W, "W"), _banks.plain(bank), ev, evec, m, lap, int(order), float(lmax), synthesis)
