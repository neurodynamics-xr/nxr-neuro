"""The Laplace-Beltrami eigenmodes of a triangle mesh (`nxr/compute.h`, `nxr::geometry`)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .. import _core


class Eigenmodes(NamedTuple):
    eigenvalues: np.ndarray    # [k], ascending
    eigenvectors: np.ndarray   # [vertices, k], the columns
    n_converged: int


def eigen(vertices, faces, k: int, sigma: float | None = None, normalize: bool = False, remove_dc: bool = False,
               intrinsic_delaunay: bool = True) -> Eigenmodes:
    """The k smallest eigenpairs of K phi = lambda M phi on a mesh (the cotangent Laplacian K and the mixed-Voronoi mass M of the operator mesh).

    Parameters
    ----------
    vertices : array [n_vertices, 3]
        Positions, float64 (other types are converted).
    faces : array [n_faces, 3]
        Vertex indices, 0-based.
    k : int
        The number of eigenpairs (1 <= k <= n_vertices - 1).
    sigma : float, optional
        Unset (the default): the scale-aware shift ladder (sigma = -1e-7 lambda_max, then -1e-4 lambda_max, then a Cholesky-mode last resort), never
        a fixed shift on the null space. A finite value is a FIXED shift-invert centre: the k eigenvalues nearest sigma are returned.
    normalize : bool
        True: M-orthonormalize the eigenvectors so that U' M U = I. Default False: as the solver produces them.
    remove_dc : bool
        True: strip the null space (every mode with |lambda| within a relative 1e-8 of 0). Default False.
    intrinsic_delaunay : bool
        True (the default): the operators are built on the intrinsic mollified Delaunay mesh. False: on the input triangulation as it is ("embedded").

    Returns
    -------
    Eigenmodes
        (eigenvalues, eigenvectors, n_converged).
    """
    return Eigenmodes(*_core.operators_eigen(vertices, faces, int(k), None if sigma is None else float(sigma), bool(normalize), bool(remove_dc),
                                                 bool(intrinsic_delaunay)))
