"""The Hodge–Dirac pencil d + delta on 0-, 1- and 2-forms of the intrinsic Delaunay triangulation (`nxr::operators::hodgedirac`)."""

from __future__ import annotations

from ... import _core
from ..geometry import _contiguous


def assemble(surface: dict, star1_floor: float = 0.0, mollify: float = 1e-6, flip_eps: float = 1e-6) -> dict:
    """The pencil (K, M), N = nV + nE + nF, rows 0-forms [0, nV), 1-forms [nV, nV + nE), 2-forms [nV + nE, N).

    K = [0 d0' *1 0; *1 d0 0 d1' *2; 0 *2 d1 0], M = diag(*0, *1, *2). The surface's own triangulation is used when it was made with these
    `mollify` and `flip_eps` (the defaults of `geometry.surfaceArrays`); otherwise the construction is run again with them.

    Parameters
    ----------
    star1_floor : float
        0 (the default): *1 is the raw cotan weight, an edge at or below the rounding level (1e-9 of the mean |weight|) raised to it. A positive value:
        every weight below it raised to it (a named option; Rheome used 1e-6), which also accepts genuinely negative weights in the mode solve.

    Returns
    -------
    dict
        ``K`` (sparse), ``M`` [N], ``d0``, ``d1`` (sparse), ``star0``, ``star1``, ``star2``, ``nV``, ``nE``, ``nF``, ``star1Floor``,
        ``star1Clamped`` (edges raised), ``star1Negative`` (of those, the genuinely negative: a boundary edge opposite an obtuse angle),
        ``eulerCharacteristic``, ``boundaryLoops``, ``genus``, ``mollifyEps``, ``faces`` [nF, 3], ``record``, ``complex`` (as `geometry.complex`).
    """
    return _core.operators_assemble(_contiguous(surface), float(star1_floor), float(mollify), float(flip_eps))
