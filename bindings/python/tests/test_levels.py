"""The levels of one manifold through the Python face (spec docs/specs/2026-11-26-levels-of-one-manifold.md): a synthetic sensor helmet (points on a spherical cap,
triangulated in the plane) and a time line. The views, their navigation, the identities of the quotient in NumPy, the product's addressing, the refusals."""

import numpy as np
import pytest
from scipy.spatial import Delaunay

from nxr_neuro.compute import geometry, operators
from nxr_neuro.compute import select as sel


@pytest.fixture(scope="module")
def helmet():
    pts = [(0.0, 0.0, 0.1)]
    for r in range(1, 10):
        th = np.deg2rad(r * 100.0 / 9.0)
        for k in range(6 * r):
            ph = 2 * np.pi * (k + 0.5 * (r % 2)) / (6 * r)
            pts.append((0.1 * np.sin(th) * np.cos(ph), 0.1 * np.sin(th) * np.sin(ph), 0.1 * np.cos(th)))
    v = np.array(pts)
    r = np.arccos(np.clip(v[:, 2] / 0.1, -1, 1))                     # the azimuthal equidistant map from the pole
    ang = np.arctan2(v[:, 1], v[:, 0])
    f = Delaunay(np.c_[r * np.cos(ang), r * np.sin(ang)]).simplices.astype(np.int32)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    flip = np.einsum("ij,ij->i", n, v[f].mean(axis=1)) < 0               # outward
    f[flip] = f[flip][:, [0, 2, 1]]
    s = geometry.surfaceArrays(v, f)
    dec = operators.assembleDECOperators(s)
    tree = sel.subjectTree(v, f)
    return v, f, dec, tree


def fine_of(dec):
    return dec["d0"], dec["hodge0"].diagonal(), dec["hodge1"].diagonal(), dec["d1"]


def dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m)


def P_of(level, V):
    P = np.zeros((V, level.cells))
    for v, c in enumerate(level.cellOf):
        if c >= 0:
            P[v, c] = 1
    return P


def test_levels_are_quotients(helmet):
    v, f, dec, tree = helmet
    d0, s0, s1, d1 = fine_of(dec)
    D0 = dense(d0)
    Lf = D0.T @ np.diag(s1) @ D0
    F = np.random.default_rng(3).standard_normal(d0.shape[0])
    for l in range(tree["depth"] + 1):
        L = sel.dyadicLevel(d0, s0, s1, tree["codes"], tree["depth"], tree["slots"], l, d1=d1)
        P = P_of(L, len(v))
        assert L.ladder == "dyadic" and L.level == l and L.cells == tree["slots"] << l
        np.testing.assert_allclose(D0 @ P, dense(L.crossing) @ dense(L.d0), atol=1e-12)        # the cochain map
        np.testing.assert_allclose(P.T @ (D0.T @ F), dense(L.d0).T @ (dense(L.crossing).T @ F), atol=1e-9)   # the boundary on the crossing edges
        np.testing.assert_allclose(P.T @ Lf @ P, dense(L.laplacian()), atol=1e-9)            # the projected Laplacian
        np.testing.assert_allclose(P.T @ s0, L.star0, atol=1e-15)
        if L.valid:
            for h in range(3 * L.faces.shape[0]):
                t = L.twin(h)
                if t >= 0:
                    assert L.twin(t) == h and L.tail(t) == L.tip(h)
                assert L.next(L.next(L.next(h))) == h
            assert L.euler == 1                                                                   # a disc
        else:
            assert L.why
            with pytest.raises(ValueError, match="no halfedges"):
                L.twin(0)


def test_navigation_and_reductions(helmet):
    v, f, dec, tree = helmet
    d0, s0, s1, d1 = fine_of(dec)
    D = tree["depth"]
    Lv = sel.vertexLevel(d0, s0, s1, d1=d1, codes=tree["codes"], depth=D, slots=tree["slots"])
    L2 = sel.dyadicLevel(d0, s0, s1, tree["codes"], D, tree["slots"], 2, d1=d1)
    L3 = sel.dyadicLevel(d0, s0, s1, tree["codes"], D, tree["slots"], 3, d1=d1)
    assert Lv.level == sel.kVertexLevel and Lv.parent(5) == (D, int(tree["codes"][5]))
    assert L3.parent(6) == (2, 3) and L2.children(3) == (3, [6, 7])
    assert sel.dyadicLevel(d0, s0, s1, tree["codes"], D, tree["slots"], 0).parent(0) is None
    for c in range(L3.cells):
        for n in L3.neighbours(c):
            assert c in L3.neighbours(n)
    assert L3.area(1) == L3.star0[1] and np.array_equal(L3.star(1), L3.star1)
    x = np.random.default_rng(4).standard_normal((2, len(v)))
    np.testing.assert_allclose(x @ Lv.reductionOperator(L3, "sum").T.toarray(), Lv.reduce(L3, "sum", x), atol=1e-12)
    mean = Lv.reduce(L3, "mean", x)
    keep = L3.star0 > 0
    np.testing.assert_allclose((x @ Lv.reductionOperator(L3, "mean").T.toarray())[:, keep], mean[:, keep], atol=1e-12)
    assert np.all(Lv.reduce(L3, "max", x)[:, keep] >= mean[:, keep])
    y = np.arange(L3.cells, dtype=float)
    assert np.array_equal(L3.expansionOperator(Lv, "mean") @ y, y[np.array(Lv.cellMap(L3))])
    assert abs((L3.expansionOperator(Lv, "sum") @ y).sum() - y[keep].sum()) < 1e-9
    with pytest.raises(ValueError, match="no expansion"):
        L3.expansionOperator(Lv, "max")
    with pytest.raises(ValueError, match="never reduced"):
        Lv.reductionOperator(L3, "none")


def test_time_line_and_product(helmet):
    v, f, dec, tree = helmet
    d0, s0, s1, d1 = fine_of(dec)
    T = sel.lineLevel(37, 200.0, 2)
    assert T.ladder == "time" and T.cells == 10 and T.neighbours(5) == [4, 6] and T.parent(9) == (3, 4)
    assert sel.lineLevel(37, 200.0, 3).children(4) == (2, [8, 9]) and T.children(9) == (1, [18])
    np.testing.assert_allclose(T.star0[-1], 1 / 200.0)
    assert T.crossingEdge(0) == 3
    S = sel.dyadicLevel(d0, s0, s1, tree["codes"], tree["depth"], tree["slots"], 3, d1=d1)
    P = sel.productLevel(S, T)
    assert P.cells == S.cells * T.cells and P.coupling == "separable"
    for c in range(P.cells):
        (ls, cs), (lt, ct) = P.jointAddress(c)
        assert (ls, lt) == (3, 2) and P.jointCode(cs, ct) == c
    c = P.jointCode(2, 4)
    assert set(P.neighbours(c)) == {P.jointCode(n, 4) for n in S.neighbours(2)} | {P.jointCode(2, 3), P.jointCode(2, 5)}
    o = P.operators()
    assert o["d0"].shape == (S.edges.shape[0] * T.cells + S.cells * (T.cells - 1), P.cells)
    with pytest.raises(ValueError, match="dispersion"):
        sel.productLevel(S, T, {"kind": "dispersion", "speed": 3.0})
