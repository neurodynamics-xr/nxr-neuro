"""The operators of a surface (`nxr/operators/dec.h`, `dirac.h`, `eigen.h`; `nxr::operators`).

The surface is the dict made by `geometry.surfaceArrays` (the operator mesh: the intrinsic Delaunay triangulation by default). Sparse matrices come back as
`scipy.sparse` CSC matrices; their edges and faces are the operator mesh's (`geometry.complex`), vertices the input's. `hodgedirac.assemble` and
`hodgeDiracModes`: the Hodge–Dirac pencil and its modes; `eigen`: the Laplace–Beltrami eigenmodes straight from (vertices, faces).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ... import _core
from ..geometry import _contiguous
from . import hodgedirac


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


def assembleManifoldOperators(surface: dict, mass: str = "mixedVoronoi") -> dict:
    """The scalar operators of the operator mesh.

    Parameters
    ----------
    surface : dict
        Made by `geometry.surfaceArrays`.
    mass : str
        "mixedVoronoi" (the default: the recorded mixed-Voronoi mass of the mollified input, diagonal), "barycentric" (diagonal) or "galerkin"
        (the consistent P1 mass, sparse).

    Returns
    -------
    dict
        ``cotanLaplacian`` [nV, nV] (positive semidefinite, sparse), ``mass`` [nV, nV] (sparse), ``massVariant``, ``vertexDualAreas`` [nV] (a third of
        each operator-mesh triangle's area per vertex), ``vertexNormals`` [nV, 3], ``totalArea``.
    """
    return _core.operators_assembleManifoldOperators(_contiguous(surface), str(mass))


def assembleDECOperators(surface: dict) -> dict:
    """The discrete exterior calculus of the operator mesh: ``d0`` [nE, nV], ``d1`` [nF, nE], ``hodge0`` [nV, nV], ``hodge1`` [nE, nE],
    ``hodge2`` [nF, nF], ``hodge1Inverse`` (all sparse).

    Edges are oriented tail -> tip as `geometry.complex` gives them. d0' hodge1 d0 is the cotan Laplacian; hodge0 is the mixed-Voronoi mass. hodge1 is the
    RAW cotan weight, not floored: a cocircular edge gives 0 (hodge1Inverse then has an infinite entry), and a boundary edge opposite an obtuse angle a
    negative weight.
    """
    return _core.operators_assembleDECOperators(_contiguous(surface))


_SOLVE_OPTIONS = ("shift", "sigma", "slice", "normalize", "tol", "ncv", "maxit", "limit")


def _solve_options(options: dict) -> dict:
    bad = set(options) - set(_SOLVE_OPTIONS)
    if bad:
        raise TypeError(f"unknown solve options {sorted(bad)}: use {', '.join(_SOLVE_OPTIONS)}")
    return options


def hodgeDiracModes(pencil: dict, count: int, both_signs: bool = False, **options) -> dict:
    """The Hodge–Dirac modes of a pencil made by `hodgedirac.assemble`, by the exact/coexact split (the reference method).

    exact modes (psi, d psi / mu, 0) / sqrt 2 from the Neumann ladder d0' *1 d0 (mass *0); coexact modes (0, delta chi / mu, chi) / sqrt 2 from the
    ladder (*2 d1) *1^-1 (*2 d1)' (mass *2; Dirichlet on an open boundary); mu = sqrt(lambda), merged by mu, the constants dropped.

    Refused (nxr_neuro.Error) on a mesh whose split is incomplete: anything but a closed genus-0 surface or a disc (one boundary loop), since no
    harmonic 1-forms are built; and when *1 has genuinely negative weights (``pencil["star1Negative"]`` > 0) and no floor was named in `assemble`.

    Parameters
    ----------
    count : int
        About this many modes (all under the single lambda cut; ``limit=`` keeps at most that many of the lowest).
    both_signs : bool
        Also return the -mu mirrors (1-form block negated).
    **options
        The solve's named options: ``shift`` ("absolute" | "relative_ladder"), ``sigma`` (-1e-3), ``slice`` (0), ``normalize`` (False), ``tol``,
        ``ncv``, ``maxit``, ``limit``.

    Returns
    -------
    dict
        ``eigenvalues`` mu [k] ascending, ``eigenvectors`` [nV + nE + nF, k] (rows: 0-forms, 1-forms, 2-forms), ``kind`` [k] (0 exact, 2 coexact),
        ``k``, ``nConverged``.
    """
    return _core.operators_hodgeDiracModes(pencil, int(count), bool(both_signs), _solve_options(options))


def hodgeLadders(L0, M0, S2, M2, count: int, **options) -> dict:
    """The two Hodge ladders separately: the k smallest eigenpairs of (L0, M0) (exact) and (S2, M2) (coexact), the null cut, the single lambda cut and
    the merge by mu.

    Returns ``exact`` and ``coexact`` (each ``eigenvalues`` lambda, ``eigenvectors``: psi [nV, k0], chi [nF, k2]), ``top`` (the lambda cut), and the kept
    modes merged by mu: ``mu``, ``kind`` (0 exact, 2 coexact), ``col`` (the column in its ladder), ``nExact``, ``nCoexact``. Options as
    `hodgeDiracModes`. No topology check: the matrices carry none (`hodgeDiracModes` checks the pencil).
    """
    import scipy.sparse as sp
    m = [sp.csc_matrix(a, dtype=np.float64) for a in (L0, M0, S2, M2)]
    return _core.operators_hodgeLadders(*m, int(count), _solve_options(options))


def poisson(K, M, density_vertices, density_values) -> np.ndarray:
    """Solve K phi = -M (rho - rho_bar), rho_bar the M-weighted mean of rho, rho given at `density_vertices` (0 elsewhere). [nV]."""
    import scipy.sparse as sp
    return _core.operators_poisson(sp.csc_matrix(K, dtype=np.float64), sp.csc_matrix(M, dtype=np.float64),
                                   [int(v) for v in np.atleast_1d(density_vertices)], [float(x) for x in np.atleast_1d(density_values)])
