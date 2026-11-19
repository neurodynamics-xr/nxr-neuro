"""nn.compute.geometry: surfaceArrays (the surface as data) and complex (the operator mesh in the face-corner layout). Synthetic meshes only."""

import numpy as np
import pytest

import nxr_neuro as nn

from test_manifold import icosphere


def disc(n=7):
    """A flat triangulated unit square, n x n vertices (an open surface with one boundary loop of 4 (n - 1) vertices)."""
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    v = np.stack([x.ravel(), y.ravel(), np.zeros(n * n)], axis=1)
    f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            f += [(a, b, d), (a, d, c)]
    return v, np.array(f, dtype=np.int32)


def _check_complex(c, nV):
    faces, edges, fe = c["faces"], c["edges"], c["faceEdges"]
    nF, nE = len(faces), len(edges)
    assert faces.shape == (nF, 3) and edges.shape == (nE, 2) and fe.shape == (nF, 3) and c["length"].shape == (nE,)
    assert faces.min() >= 0 and faces.max() < nV and np.all(c["length"] > 0)
    # each corner's halfedge runs faces[f, k] -> faces[f, k+1]; faceEdges names its edge and whether it runs along the edge's tail -> tip
    e = np.abs(fe) - 1
    tail, tip = faces, np.roll(faces, -1, axis=1)
    along = fe > 0
    np.testing.assert_array_equal(np.where(along, edges[e, 0], edges[e, 1]), tail)
    np.testing.assert_array_equal(np.where(along, edges[e, 1], edges[e, 0]), tip)
    return nV - nE + nF


def test_sphere_surface_and_complex():
    v, f = icosphere(2)
    s = nn.compute.geometry.surfaceArrays(v, f)
    assert s["positions"].shape == v.shape and s["faces"].shape == f.shape
    rec = s["triangulation"]["record"]
    assert rec["mesh"] == "intrinsic" and rec["delaunay"] and rec["boundaryVertices"] == 0 and rec["flips"] >= 0
    assert s["triangulation"]["mass"].shape == (len(v),)
    np.testing.assert_allclose(s["triangulation"]["mass"].sum(), 4 * np.pi, rtol=0.02)
    assert _check_complex(nn.compute.geometry.complex(s), len(v)) == 2


def test_disc_surface_has_one_boundary():
    v, f = disc()
    s = nn.compute.geometry.surfaceArrays(v, f)
    rec = s["triangulation"]["record"]
    assert rec["boundaryVertices"] == 24 and s["triangulation"]["boundaryLoops"] == 1
    assert _check_complex(nn.compute.geometry.complex(s), len(v)) == 1


def test_embedded_keeps_the_input_triangulation():
    v, f = icosphere(1)
    s = nn.compute.geometry.surfaceArrays(v, f, intrinsic_delaunay=False)
    assert s["triangulation"]["record"]["mesh"] == "embedded"
    c = nn.compute.geometry.complex(s)
    # the same triangles, each possibly started at another corner
    canon = lambda t: {tuple(np.roll(r, -int(np.argmin(r)))) for r in t}
    assert canon(c["faces"]) == canon(f)


def test_complex_is_the_same_from_a_copied_surface():
    # the surface is data: a deep copy (as after storing it) gives the same operator mesh
    v, f = icosphere(2)
    s = nn.compute.geometry.surfaceArrays(v, f)
    t = {"positions": s["positions"].copy(), "faces": s["faces"].copy(),
         "triangulation": {k: (a.copy() if isinstance(a, np.ndarray) else a) for k, a in s["triangulation"].items()}}
    a, b = nn.compute.geometry.complex(s), nn.compute.geometry.complex(t)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_refusals():
    v, f = icosphere(1)
    with pytest.raises(ValueError):
        nn.compute.geometry.surfaceArrays(v[:, :2], f)
    with pytest.raises(nn.Error):
        nn.compute.geometry.surfaceArrays(v, np.vstack([f, [[0, 1, len(v)]]]))
    with pytest.raises(TypeError):
        nn.compute.geometry.complex({"positions": v})
