"""Writes integrals_golden.txt: the reference of the tile integrals (a flow's flux and circulation over tiles) — nxr's `measure.integrals` (`flux`, `circulation`, `face_field`, `vertex_area`)
on SYNTHETIC meshes: a closed icosphere (a field of three frames, tiles by octant) and a folded, jittered plane (tiles by region), exact to rounding.
Run:  uv run --project <nxr checkout> python compute/test/golden/gen_integrals_golden.py compute/test/golden/integrals_golden.txt"""
import sys

import numpy as np
from nxr.measure.integrals import circulation, face_field, flux, vertex_area
from nxr.operators import face_gradient

R = lambda v: repr(float(v))
out = open(sys.argv[1], "w")


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f, dtype=np.int32)


def folded(n=24, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    V = np.c_[x.ravel(), y.ravel(), np.zeros(n * n)]
    inner = (V[:, 0] > 0) & (V[:, 0] < 1) & (V[:, 1] > 0) & (V[:, 1] < 1)
    V[inner, :2] += rng.uniform(-0.3, 0.3, (inner.sum(), 2)) / n
    V[:, 2] = 0.08 * np.sin(14 * V[:, 0]) * np.cos(11 * V[:, 1])
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, i * n + j + 1, (i + 1) * n + j + 1
            F += [[a, b, d], [a, d, c]]
    return V, np.array(F, dtype=np.int32)


def case(tag, V, F, J, codes):
    fg = face_gradient(V, F)
    phi, gam = flux(J, fg, codes), circulation(J, fg, codes)
    out.write(f"case {tag} {len(V)} {len(F)} {J.shape[1]} {len(phi.codes)}\n")
    out.write("P " + " ".join(R(v) for v in V.ravel()) + "\n")
    out.write("F " + " ".join(str(int(v)) for v in F.ravel()) + "\n")
    out.write("J " + " ".join(R(v) for v in J.ravel()) + "\n")           # [3nV x T] row-major
    out.write("codes " + " ".join(str(int(v)) for v in codes) + "\n")
    out.write("vertex_area " + " ".join(R(v) for v in vertex_area(fg)) + "\n")
    out.write("face_field " + " ".join(R(v) for v in face_field(J, fg).ravel()) + "\n")   # [nF x 3 x T]
    out.write("tile_codes " + " ".join(str(int(v)) for v in phi.codes) + "\n")
    out.write("flux " + " ".join(R(v) for v in phi.value.ravel()) + "\n")           # [tiles x T]
    out.write("circulation " + " ".join(R(v) for v in gam.value.ravel()) + "\n")
    out.write("area " + " ".join(R(v) for v in phi.area) + "\n")


V, F = icosphere(3)
V = V * 0.07
rng = np.random.default_rng(3)
J = np.stack([(np.cross(V, [0.2, 0.1, 1.0]) + np.c_[np.sin(40 * V[:, 0]), 0 * V[:, 0], V[:, 2]]).ravel() * (1 + 0.3 * t) + 0.01 * rng.standard_normal(3 * len(V)) for t in range(3)], 1)
codes = (V[:, 2] > 0).astype(int) + 2 * (V[:, 0] > 0.02) + 4 * (V[:, 1] > 0.0)
case("sphere", V, F, J, codes)
V, F = folded()
Jf = np.stack([np.c_[np.sin(3 * V[:, 0]) + V[:, 1] ** 2, np.cos(2 * V[:, 1]) - V[:, 0] * V[:, 1], np.sin(5 * V[:, 0])].ravel() * (1 + t) for t in range(2)], 1)
codes = np.where(np.hypot(V[:, 0] - 0.45, V[:, 1] - 0.55) < 0.3, 0, np.where(V[:, 0] < 0.2, 1, -1))     # a disc, a strip, the rest off every tile
case("folded", V, F, Jf, codes)
out.close()
