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
    ic = S.intrinsicComplex(s)
    J = np.tile([0.3, -0.7, 0.0], (len(v), 1))
    w = S.flat(ic, J)
    c = G.complex(s)
    np.testing.assert_allclose(w, (v[c["edges"][:, 1]] - v[c["edges"][:, 0]]) @ J[0], atol=1e-6)
    np.testing.assert_allclose(S.sharp(ic, w), J, atol=1e-6)
    W = np.stack([w, 2 * w], axis=1)
    assert S.sharp(ic, W).shape == (len(v), 3, 2) and S.flat(ic, np.stack([J, J], axis=2)).shape == (len(c["edges"]), 2)


def test_forms_on_the_modes(sphere):
    v, _, s = sphere
    h = O.hodgedirac.assemble(s)
    md = S.modesOf(O.hodgeDiracModes(h, 40, limit=30), h)
    ic = S.intrinsicComplex(s)
    assert ic["nE"] == len(G.complex(s)["edges"]) and ic["fanStart"].shape == (len(v) + 1,)
    assert md["P"].shape == (len(v), 30) and set(md["kind"]) == {0, 2}
    rng = np.random.default_rng(0)
    c = rng.standard_normal((30, 2))
    w = S.synthesizeOneForm(ic, md, c)
    np.testing.assert_allclose(S.projectOneForm(ic, md, w), c, atol=1e-8)
    ce = np.where((md["kind"] == 0)[:, None], c, 0)
    d0 = O.assembleDECOperators(s)["d0"]
    np.testing.assert_allclose(d0 @ S.exactPotential(md, c), S.synthesizeOneForm(ic, md, ce), atol=1e-10)
    sp = S.splitPhase(ic, md, w[:, 0])
    np.testing.assert_allclose(S.coexactStream(ic, md, c[:, 0]), sp["stream"], atol=1e-12)
    assert sp["propagation"].shape == (len(v), 3) and "shareTiming" not in sp and "inconsistency" not in sp
    # the energies are pointwise: the vertices' values sum to the *1 energy (a share is a ratio of such sums, derived at read)
    star1 = np.asarray(ic["star1"])
    ev = S.vertexEnergy(ic, np.stack([w[:, 0], sp["exact"]], axis=1))
    assert ev.shape == (len(v), 2)
    np.testing.assert_allclose(ev.sum(axis=0), [np.sum(star1 * w[:, 0] ** 2), np.sum(star1 * sp["exact"] ** 2)], rtol=1e-12)
    # the phase current against its formula, one column a frame
    z = np.exp(1j * 3 * v[:, 0]) * (1 + 0.5 * v[:, 2])
    e = G.complex(s)["edges"]
    J = S.phaseCurrent(ic, np.stack([z, 2 * z], axis=1))
    want = np.imag(np.conj(z[e[:, 0]]) * z[e[:, 1]])
    np.testing.assert_allclose(J[:, 0], want, atol=1e-14)
    np.testing.assert_allclose(J[:, 1], 4 * want, atol=1e-13)
    # modes of another triangulation are refused
    v1, f1 = icosphere(1)
    with pytest.raises(ValueError, match="same triangulation"):
        S.projectOneForm(S.intrinsicComplex(G.surfaceArrays(v1, f1)), md, np.zeros(120))
    with pytest.raises(TypeError, match="intrinsicComplex"):
        S.projectOneForm(s, md, w)


def test_plane_wave_on_a_disc_is_timing():
    # a plane wave on a flat disc (open, one loop): its phase steps are a gradient, read as timing with no rotation; what a truncated basis misses is residual
    # (a linear phase converges slowly in the Neumann modes), and the potential follows the phase
    v, f = disc(11)
    s = G.surfaceArrays(v, f)
    h = O.hodgedirac.assemble(s)
    ic = S.intrinsicComplex(s)
    phase = 2.0 * v[:, 0] + 1.0 * v[:, 1]
    w = S.phaseGradient(ic, np.exp(1j * phase))
    for count in (40, 80):
        sp = S.splitPhase(ic, S.modesOf(O.hodgeDiracModes(h, count), h), w)
        energy, timing, rotation = S.vertexEnergy(ic, np.stack([w, sp["exact"], sp["coexact"]], axis=1)).sum(axis=0)   # the sums over the whole disc
        assert rotation / energy < 1e-12 and 0.9 < timing / energy < 1
        assert np.corrcoef(sp["potential"], phase)[0, 1] > 0.997


def test_phase_current_one_quantity_two_discretisations():
    # the same quantity whichever operator is passed: on a fine flat disc, flat(the vertex current) ~ the edge current, and the edge phase gradient is the
    # vertex one integrated along each edge
    v, f = disc(21)
    s = G.surfaceArrays(v, f)
    ic = S.intrinsicComplex(s)
    phase = 3.0 * v[:, 0] - 1.5 * v[:, 1]
    z = (1 + 0.3 * v[:, 0]) * np.exp(1j * phase)
    n = np.tile([0.0, 0.0, 1.0], (len(v), 1))
    j = S.phaseCurrent(S.vertexGradient(v, f), z, normals=n)
    J = S.phaseCurrent(ic, z)
    assert j.shape == (len(v), 3) and J.shape == (ic["nE"],)
    assert np.abs(S.flat(ic, j) - J).max() < 0.02 * np.abs(J).max()
    g = S.phaseGradient(j, np.abs(z) ** 2)
    e = ic["edges"]
    np.testing.assert_allclose(S.phaseGradient(ic, z), (v[e[:, 1]] - v[e[:, 0]]) @ [3.0, -1.5, 0.0], atol=1e-10)
    np.testing.assert_allclose(g[:, :2], np.tile([3.0, -1.5], (len(v), 1)), rtol=0.03)   # the vertex (P1) discretisation is approximate; the edge one exact


def test_log_map(sphere):
    # on the unit sphere the log map's norm is the geodesic distance (the heat distance's, to its accuracy) and the frame is orthonormal and tangent
    v, _, s = sphere
    r = G.logMap(s, 0)
    d = np.linalg.norm(r["logCoords"], axis=1)
    assert r["logCoords"].shape == (len(v), 2) and d[0] < 1e-9
    true = np.arccos(np.clip(v @ v[0], -1, 1))
    near = true < 1.0
    np.testing.assert_allclose(d[near], true[near], atol=0.03)
    e1, e2, n = r["sourceE1"], r["sourceE2"], r["sourceNormal"]
    np.testing.assert_allclose([e1 @ e1, e2 @ e2, e1 @ e2, e1 @ v[0]], [1, 1, 0, 0], atol=1e-9)
    assert n @ v[0] > 0.99
    # near the source the coordinates point the right way: source + x e1 + y e2 ~ the vertex
    k = np.argsort(np.where(near, true, np.inf))[1:6]
    approx = v[0] + r["logCoords"][k, :1] * e1 + r["logCoords"][k, 1:] * e2
    assert np.abs(approx - v[k]).max() < 0.05
    both = G.logMap(s, [0, 5])
    assert len(both) == 2
    np.testing.assert_allclose(both[0]["logCoords"], r["logCoords"], atol=1e-10)


def test_log_map_face_point():
    # a point of an input face: the blend of its corners' log maps (first order). On the unit sphere, at a face's centroid the norm of the coordinates is
    # the great-circle distance from the point, better than the nearest vertex's log map taken as the point's
    v, f = icosphere(3)
    s = G.surfaceArrays(v, f)
    fi = 7
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    r = G.logMap(s, (fi, tuple(w)))
    p = w @ v[f[fi]]
    assert r["logCoords"].shape == (len(v), 2)
    np.testing.assert_allclose(r["sourcePoint"], p, atol=1e-12)
    e1, e2, n = r["sourceE1"], r["sourceE2"], r["sourceNormal"]
    np.testing.assert_allclose([e1 @ e1, e2 @ e2, e1 @ e2, e1 @ n, e2 @ n], [1, 1, 0, 0, 0], atol=1e-9)
    pu = p / np.linalg.norm(p)
    true = np.arccos(np.clip(v @ pu, -1, 1))
    near = true < 1.0
    d = np.linalg.norm(r["logCoords"], axis=1)
    err_face = np.abs(d[near] - true[near])
    np.testing.assert_allclose(d[near], true[near], atol=0.03)
    # the nearest vertex's log map, read as the log map of the point
    k = f[fi][np.argmin(np.linalg.norm(v[f[fi]] - p, axis=1))]
    dv = np.linalg.norm(G.logMap(s, int(k))["logCoords"], axis=1)
    err_vertex = np.abs(dv[near] - true[near])
    assert err_face.mean() < err_vertex.mean() and err_face.max() < err_vertex.max()
    # at a corner: the vertex log map, coordinates identical
    c = G.logMap(s, (fi, (1.0, 0.0, 0.0)))
    np.testing.assert_allclose(c["logCoords"], G.logMap(s, int(f[fi][0]))["logCoords"], atol=1e-9)
    # a list of points: one solver, same results
    both = G.logMap(s, [(fi, (1.0, 0.0, 0.0)), (fi, tuple(w))])
    assert len(both) == 2
    np.testing.assert_allclose(both[1]["logCoords"], r["logCoords"], atol=1e-10)
    for bad in [(fi, (0.5, 0.6, -0.1)), (fi, (0.5, 0.3, 0.3)), (len(f), (1.0, 0.0, 0.0))]:
        with pytest.raises(Exception):
            G.logMap(s, bad)
    with pytest.raises(ValueError):
        G.logMap(s, [0, (fi, tuple(w))])
