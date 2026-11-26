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


def meshTopology(surface: dict) -> dict:
    """The INPUT mesh's halfedge topology as arrays (geometry-central's ids, 0-based; -1 = none or exterior).

    Returns ``nV, nE, nF, nH, nC`` and, int64: ``vertexHalfedge``, ``edgeHalfedge``, ``faceHalfedge``, ``cornerHalfedge``, ``heTwin``, ``heNext``,
    ``heVertex`` (tail), ``heEdge``, ``heFace`` (-1 for exterior halfedges), ``heCorner``; uint8: ``heOrientation`` (1 when the halfedge runs along
    its edge), ``heIsInterior``. The boundary edges are ``heEdge[heIsInterior == 0]``, the boundary vertices ``heVertex[heIsInterior == 0]``;
    the number of boundary loops is ``surface["triangulation"]["boundaryLoops"]``.
    """
    return _core.geometry_meshTopology(_contiguous(surface))


def eulerCharacteristic(surface: dict) -> int:
    """chi = nV - nE + nF of the input mesh (2 - 2 genus - boundary loops)."""
    return _core.geometry_eulerCharacteristic(_contiguous(surface))


def vertexFrames(surface: dict) -> dict:
    """The tangent frame at each vertex: ``e1``, ``e2`` = n x e1, ``normals`` (corner-angle weighted), each [nV, 3].

    The frame is the gauge of the (mollified) input mesh: the tangent space the connection Laplacian's coordinates, the vector heat method and the
    transport use; a tangent vector with coordinates (a, b) is a e1 + b e2.
    """
    return _core.geometry_vertexFrames(_contiguous(surface))


def frames(surface: dict) -> dict:
    """The tangent frame of each INPUT face: ``e1``, ``e2`` = n x e1, ``normals``, each [nF, 3]."""
    return _core.geometry_frames(_contiguous(surface))


def heat(surface: dict, sources, t_coef: float = 1.0, mesh: str = "intrinsic") -> np.ndarray:
    """Geodesic distances by the heat method (Crane et al. 2013).

    Parameters
    ----------
    surface : dict
        Made by :func:`surfaceArrays`.
    sources : int, sequence of int, or sequence of sequences of int
        One source set (a vertex or a list of vertices): the distance to the nearest of them, [nV]. A list of sets: one column per set, [nV, n_sets].
    t_coef : float
        The time step as a multiple of the squared mean edge length (default 1).
    mesh : str
        "intrinsic" (the default): on the operator mesh (the intrinsic Delaunay triangulation). "input": geometry-central's heat method on the input.
    """
    if np.isscalar(sources):
        sets, single = [[int(sources)]], True
    elif len(sources) and not np.isscalar(sources[0]):
        sets, single = [[int(v) for v in s] for s in sources], False
    else:
        sets, single = [[int(v) for v in sources]], True
    D = _core.geometry_heat(_contiguous(surface), sets, float(t_coef), str(mesh))
    return D[:, 0] if single else D


def logMap(surface: dict, source, strategy: str = "affine_local", t_coef: float = 1.0):
    """The log map at a source vertex: the geodesic polar coordinates of every vertex in the source's tangent plane (geometry-central's vector heat
    method, on the operator mesh, the intrinsic Delaunay triangulation).

    Parameters
    ----------
    source : int, a sequence of int, ``(face, (w0, w1, w2))``, or a sequence of such pairs
        One source vertex (one result), or several (a list of results, solved on one solver: the factorisation is shared). A pair is a point of an INPUT
        face: its index (a row of the surface's faces) and its barycentric weights (non-negative, summing to 1; weight k belongs to corner k of the face).
        A face point's log map blends its three corners' log maps (first order in the face's size; exact at a corner); one kind of source per call.
    strategy : str
        "affine_local" (the default: the affine heat method, prefactored), "affine_adaptive" (no prefactoring, the most accurate near the source),
        or "vector_heat" (the original vector heat method).
    t_coef : float
        The diffusion time as a multiple of the squared mean edge length (default 1).

    Returns
    -------
    dict (or a list of them)
        ``logCoords`` [nV, 2] (x, y) in the source's tangent plane: the norm is the geodesic distance from the source, atan2(y, x) the direction;
        ``sourceE1``, ``sourceE2`` [3] the plane's frame in R^3 (the gauge of `vertexFrames`); ``sourceNormal`` [3] = e1 x e2. A point near the source is
        approximately source + x e1 + y e2.
        For a face point the frame is the heaviest corner's, rotated into the face's plane (``sourceNormal`` is the face's unit normal), and
        ``sourcePoint`` [3] is the point in R^3 (the weighted sum of the corners).
    """
    def is_point(x):
        return not np.isscalar(x) and len(x) == 2 and np.isscalar(x[0]) and not np.isscalar(x[1])
    single = np.isscalar(source) or is_point(source)
    items = [source] if single else list(source)
    pts = [is_point(x) for x in items]
    if any(pts) and not all(pts):
        raise ValueError("logMap: the sources are vertices or (face, weights) pairs, not both")
    if items and all(pts):
        faces = [int(f) for f, _ in items]
        bary = np.array([np.asarray(w, dtype=float).reshape(3) for _, w in items]).reshape(-1, 3)
        out = _core.geometry_logMap(_contiguous(surface), [], faces, bary, str(strategy), float(t_coef))
    else:
        out = _core.geometry_logMap(_contiguous(surface), [int(v) for v in items], [], np.zeros((0, 3)), str(strategy), float(t_coef))
        for r in out:
            r["sourceNormal"] = np.cross(r["sourceE1"], r["sourceE2"])
    return out[0] if single else out
