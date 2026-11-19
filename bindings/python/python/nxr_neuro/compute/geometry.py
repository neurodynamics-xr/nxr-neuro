"""The surface as data (`nxr/geometry/surface_arrays.h`, `nxr/geometry/intrinsic_surface.h`, `nxr::geometry`).

`surfaceArrays` makes the surface once: the input mesh and its intrinsic mollified Delaunay triangulation (the expensive step: mollification and edge
flips), as a dict of NumPy arrays. Every surface function takes that dict; nothing is cached between calls. The triangulation's arrays are
geometry-central's own layout: pass the dict on, do not edit it. `complex` reads the operator mesh from it in the face-corner layout.
"""

from __future__ import annotations

import numpy as np

from .. import _core


def surfaceArrays(vertices, faces, intrinsic_delaunay: bool = True, mollify: float = 1e-6, flip_eps: float = 1e-6, surface: str = "") -> dict:
    """The surface arrays of a mesh: its positions and faces, and the intrinsic triangulation every operator is built on.

    Parameters
    ----------
    vertices : array [n_vertices, 3]
        Positions (converted to float64).
    faces : array [n_faces, 3]
        Vertex indices, 0-based, consistently oriented (converted to int32).
    intrinsic_delaunay : bool
        True (the default): the intrinsic mollified Delaunay triangulation (geometry-central's signpost triangulation). False: the named option
        `mesh = embedded`, the input triangulation as it is (no mollification, no flips).
    mollify : float
        The mollification margin delta, relative to the mean input edge length (default 1e-6; 0 = no margin, epsilon is still added where a triangle
        is invalid).
    flip_eps : float
        An edge is flipped while its cotangent weight is below -flip_eps (default 1e-6).
    surface : str
        The surface's id, recorded in the gauge identity ("" when there is none).

    Returns
    -------
    dict
        ``positions`` [nV, 3] float64, ``faces`` [nF, 3] int32 (the input), and ``triangulation``: ``options`` and ``record`` (what the construction
        was told and did: ``mesh``, ``mollify``, ``epsilon`` (the length added to every input edge), ``flipEps``, ``flips``, ``delaunay``, ``mass``,
        ``boundaryVertices``, ``geometryCentral``, ``fingerprint``, ``gauge``, ``gaugeId``), ``mass`` [nV] (the lumped mass the operators use),
        ``inputEdgeLength``, ``edgeLength`` [nE] (intrinsic), and geometry-central's halfedge arrays.

    Raises
    ------
    nxr_neuro.Error
        A face index outside [0, nV), a non-triangle, non-manifold or inconsistently oriented mesh, or a non-finite or non-positive edge length.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("vertices and faces are [n, 3]")
    return _core.geometry_surfaceArrays(v, f, bool(intrinsic_delaunay), float(mollify), float(flip_eps), str(surface))


def complex(surface: dict) -> dict:  # noqa: A001 (the C++ word: IntrinsicSurface::complex)
    """The operator mesh of a surface (the intrinsic triangulation; with `mesh = embedded` the input) in the face-corner layout.

    Face f has the corners 3f + k (k = 0, 1, 2); the halfedge at corner (f, k) runs from ``faces[f, k]`` to ``faces[f, (k + 1) % 3]``. Edges are
    numbered as the triangulation numbers them and oriented tail -> tip. The intrinsic faces and edges can differ from the input's (edge flips).

    Parameters
    ----------
    surface : dict
        Made by :func:`surfaceArrays`.

    Returns
    -------
    dict
        ``faces`` [nF, 3] int32, the vertex at each corner; ``edges`` [nE, 2] int32, (tail, tip); ``faceEdges`` [nF, 3] int32, the signed 1-based
        edge of the halfedge at each corner (+(e + 1) along the edge's orientation, -(e + 1) against it); ``length`` [nE] the intrinsic edge
        lengths; ``angle`` [nF, 3] the raw signpost angle of each corner's halfedge, from the input's reference halfedge; ``angleSum`` [nV] the cone
        angle; ``reference`` [nV] int32, the vertex at angle 0 (the head of the input's reference halfedge).
    """
    return _core.geometry_complex(_contiguous(surface))


_INT_FIELDS = ("halfedgeNext", "halfedgeVertex", "halfedgeFace", "vertexHalfedge", "faceHalfedge")


def _contiguous(surface: dict) -> dict:
    """The surface dict with every array C-contiguous in the layout the core reads (a no-op for a dict made by surfaceArrays)."""
    if not isinstance(surface, dict) or "triangulation" not in surface:
        raise TypeError("expected the surface dict made by geometry.surfaceArrays(vertices, faces)")
    t = dict(surface["triangulation"])
    for k, a in t.items():
        if isinstance(a, np.ndarray):
            t[k] = np.ascontiguousarray(a, dtype=np.int64 if k in _INT_FIELDS else np.float64)
    return {"positions": np.ascontiguousarray(surface["positions"], dtype=np.float64),
            "faces": np.ascontiguousarray(surface["faces"], dtype=np.int32), "triangulation": t}
