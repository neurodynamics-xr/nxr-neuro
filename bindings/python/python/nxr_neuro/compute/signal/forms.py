"""1-forms on the intrinsic complex and tangent vector fields (`nxr/signal/intrinsic_forms.h`, `nxr::signal`)."""

from __future__ import annotations

import numpy as np

from ... import _core
from ..geometry import _contiguous


def sharp(surface: dict, w) -> np.ndarray:
    """A 1-form on the intrinsic edges -> a tangent vector at each vertex, in R^3.

    At each vertex the least-squares tangent vector v with v . t_e = w_e / L_e over its edges, t_e the edge's direction at the vertex (its signpost
    angle placed between the input edges that bracket it). Edges as `geometry.complex` gives them (tail -> tip).

    Parameters
    ----------
    w : array [nE] or [nE, m]
        The 1-form (m of them as columns).

    Returns
    -------
    array [nV, 3] or [nV, 3, m]
    """
    w = np.asarray(w, dtype=np.float64)
    single = w.ndim == 1
    J = _core.signal_sharp(_contiguous(surface), w.reshape(len(w), -1))
    J = np.asarray(J).reshape(-1, 3, J.shape[1])
    return J[:, :, 0] if single else J


def flat(surface: dict, J) -> np.ndarray:
    """A vertex vector field in R^3 -> a 1-form on the intrinsic edges: w_e = 1/2 L_e (J_tail . t_tail + J_tip . t_tip) (the field integrated along the
    edge, trapezoid rule; t_e the edge's direction at each end, tail -> tip).

    Parameters
    ----------
    J : array [nV, 3] or [nV, 3, m]

    Returns
    -------
    array [nE] or [nE, m]
    """
    J = np.asarray(J, dtype=np.float64)
    single = J.ndim == 2
    if J.shape[1] != 3:
        raise ValueError("the vector field is [n_vertices, 3] or [n_vertices, 3, m]")
    w = _core.signal_flat(_contiguous(surface), J.reshape(J.shape[0] * 3, -1))
    return np.asarray(w)[:, 0] if single else np.asarray(w)
