"""nn.compute.operators and the surface functions built on geometry.surfaceArrays: the scalar and DEC operators, the Hodge–Dirac pencil and its modes
(with the checks that refuse an incomplete split), the ladders, Poisson, frames, the heat method, and sharp/flat. Synthetic meshes only."""

import numpy as np
import pytest

import nxr_neuro as nn

from test_geometry import disc
from test_manifold import icosphere

G, O, S = nn.compute.geometry, nn.compute.operators, nn.compute.signal


def annulus(n=9):
    """The flat n x n grid with its centre cell removed: two boundary loops."""
    v, f = disc(n)
    c = (n - 1) // 2
    a = c * n + c
    keep = [not (set(t) <= {a, a + 1, a + n, a + n + 1}) for t in f]
    return v, f[keep]


@pytest.fixture(scope="module")
def sphere():
    v, f = icosphere(2)
    return v, f, G.surfaceArrays(v, f)


def test_topology(sphere):
    _, _, s = sphere
    assert G.eulerCharacteristic(s) == 2
    v, f = disc(7)
    d = G.surfaceArrays(v, f)
    t = G.meshTopology(d)
    assert G.eulerCharacteristic(d) == 1 and t["nV"] == 49
    assert len(np.unique(t["heEdge"][t["heIsInterior"] == 0])) == 24 and len(np.unique(t["heVertex"][t["heIsInterior"] == 0])) == 24


def test_manifold_operators(sphere):
    v, f, s = sphere
    o = O.assembleManifoldOperators(s)
    K, M = o["cotanLaplacian"], o["mass"]
    assert K.shape == (len(v), len(v)) and abs(K - K.T).max() < 1e-12
    assert np.abs(K @ np.ones(len(v))).max() < 1e-10
    np.testing.assert_allclose(M.diagonal().sum(), o["totalArea"], rtol=1e-10)
    np.testing.assert_allclose(M.diagonal(), s["triangulation"]["mass"], rtol=1e-12)
    assert o["vertexNormals"].shape == (len(v), 3)
    # the same pencil as operators.eigen's
    r = O.eigen(v, f, 10)
    phi = r.eigenvectors[:, 4]
    np.testing.assert_allclose(K @ phi, r.eigenvalues[4] * (M @ phi), atol=1e-8 * np.abs(K @ phi).max())


def test_dec_operators(sphere):
    _, _, s = sphere
    d = O.assembleDECOperators(s)
    c = G.complex(s)
    assert abs(d["d1"] @ d["d0"]).max() < 1e-14
    L = (d["d0"].T @ d["hodge1"] @ d["d0"]).toarray()
    np.testing.assert_allclose(L, O.assembleManifoldOperators(s)["cotanLaplacian"].toarray(), atol=1e-12)
    # d0's rows are the complex's edges, tail -> tip
    e = np.arange(len(c["edges"]))
    D = d["d0"].tocsr()
    np.testing.assert_array_equal(D[e, c["edges"][:, 1]].A1, 1)
    np.testing.assert_array_equal(D[e, c["edges"][:, 0]].A1, -1)


def test_hodge_dirac_sphere(sphere):
    _, _, s = sphere
    h = O.hodgedirac.assemble(s)
    assert (h["eulerCharacteristic"], h["boundaryLoops"], h["genus"], h["star1Negative"]) == (2, 0, 0, 0)
    N = h["nV"] + h["nE"] + h["nF"]
    assert h["K"].shape == (N, N) and h["M"].shape == (N,)
    m = O.hodgeDiracModes(h, 20, limit=6)
    assert m["eigenvectors"].shape == (N, 6) and set(m["kind"]) <= {0, 2}
    np.testing.assert_allclose(m["eigenvalues"][0] ** 2, 2.0, rtol=0.02)     # l(l+1), l = 1
    # each mode solves the pencil: K x = mu M x
    x, mu = m["eigenvectors"][:, 0], m["eigenvalues"][0]
    np.testing.assert_allclose(h["K"] @ x, mu * h["M"] * x, atol=1e-6 * np.abs(h["K"] @ x).max())


def test_hodge_dirac_disc_and_ladders():
    v, f = disc(9)
    s = G.surfaceArrays(v, f)
    h = O.hodgedirac.assemble(s)
    assert (h["eulerCharacteristic"], h["boundaryLoops"], h["genus"]) == (1, 1, 0)
    m = O.hodgeDiracModes(h, 24)
    # the ladders give the same modes: Neumann exact (a zero mode, dropped), Dirichlet coexact (no zero mode)
    import scipy.sparse as sp
    s1 = sp.diags(h["star1"])
    B = sp.diags(h["star2"]) @ h["d1"]
    lad = O.hodgeLadders(h["d0"].T @ s1 @ h["d0"], sp.diags(h["star0"]), B @ sp.diags(1 / h["star1"]) @ B.T, sp.diags(h["star2"]), 24)
    assert abs(lad["exact"]["eigenvalues"][0]) < 1e-8 and lad["coexact"]["eigenvalues"][0] > 1.0
    np.testing.assert_allclose(lad["mu"], m["eigenvalues"], rtol=1e-8)
    np.testing.assert_array_equal(lad["kind"], m["kind"])


def test_hodge_dirac_refusals():
    v, f = annulus()
    h = O.hodgedirac.assemble(G.surfaceArrays(v, f))
    assert h["boundaryLoops"] == 2
    with pytest.raises(nn.Error, match="harmonic"):
        O.hodgeDiracModes(h, 10)
    t = O.hodgedirac.assemble(G.surfaceArrays(np.array([[0, 0, 0], [2, 0, 0], [1, 0.2, 0]], float), np.array([[0, 1, 2]])))
    assert t["star1Negative"] == 1
    with pytest.raises(nn.Error, match="negative cotan weight"):
        O.hodgeDiracModes(t, 1)
    with pytest.raises(TypeError):
        O.hodgeDiracModes(h, 10, sigmaa=1.0)


def test_poisson(sphere):
    v, _, s = sphere
    o = O.assembleManifoldOperators(s)
    phi = O.poisson(o["cotanLaplacian"], o["mass"], [0, len(v) - 1], [1.0, -1.0])
    rho = np.zeros(len(v)); rho[0], rho[-1] = 1.0, -1.0
    Mrho = o["mass"] @ rho
    rhs = -(Mrho - o["mass"].diagonal() * (Mrho.sum() / o["mass"].diagonal().sum()))
    np.testing.assert_allclose(o["cotanLaplacian"] @ phi, rhs, atol=1e-6 * np.abs(rhs).max())


def test_frames(sphere):
    v, f, s = sphere
    for fr, n in ((G.vertexFrames(s), len(v)), (G.frames(s), len(f))):
        e1, e2, nn_ = fr["e1"], fr["e2"], fr["normals"]
        assert e1.shape == (n, 3)
        np.testing.assert_allclose(np.linalg.norm(e1, axis=1), 1, atol=1e-10)
        np.testing.assert_allclose(np.einsum("ij,ij->i", e1, nn_), 0, atol=1e-10)
        np.testing.assert_allclose(np.cross(nn_, e1), e2, atol=1e-10)
    # vertex normals on the unit sphere point outward
    assert np.einsum("ij,ij->i", G.vertexFrames(s)["normals"], v).min() > 0.99


def test_heat(sphere):
    v, _, s = sphere
    far = int(np.argmin(v @ v[0]))
    d = G.heat(s, 0)
    assert d.shape == (len(v),) and d[0] < 1e-6
    np.testing.assert_allclose(d[far], np.pi, rtol=0.05)
    D = G.heat(s, [[0], [far]])
    assert D.shape == (len(v), 2)
    np.testing.assert_allclose(D[:, 0], d, rtol=1e-10, atol=1e-12)


def test_sharp_and_flat_on_a_flat_disc():
    # on a flat mesh a constant field is integrated exactly along every edge and recovered exactly from its 1-form
    v, f = disc(7)
    s = G.surfaceArrays(v, f)
    J = np.tile([0.3, -0.7, 0.0], (len(v), 1))
    w = S.flat(s, J)
    c = G.complex(s)
    np.testing.assert_allclose(w, (v[c["edges"][:, 1]] - v[c["edges"][:, 0]]) @ J[0], atol=1e-6)
    np.testing.assert_allclose(S.sharp(s, w), J, atol=1e-6)
    W = np.stack([w, 2 * w], axis=1)
    assert S.sharp(s, W).shape == (len(v), 3, 2) and S.flat(s, np.stack([J, J], axis=2)).shape == (len(c["edges"]), 2)
