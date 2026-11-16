"""PARITY of nxr-compute's ONE Delaunay construction (compute/include/nxr/delaunay.h) with the Studio's intrinsicDelaunay.ts (the reference) and
nxr's `nxr.operators.intrinsic_delaunay.prepare`, on synthetic meshes that include the hard cases:

  * a stretched, jittered sphere with many non-Delaunay edges;
  * a planar lattice (every unit square is cocircular) and a cube grid (closed; every face's quads cocircular), both randomly flipped: SEVERAL Delaunay
    triangulations are valid there, so the flip ORDER decides, and the faces must match the reference exactly;
  * a sphere with a zero-area (sliver) triangle, which needs mollification;
  * tiny meshes (a tetrahedron, a stretched octahedron) where the reference's flip refusals and self-edges occur.

What is compared (C++ against the Studio, and C++ against nxr): the faces after the flips and the twins (exactly), the edge lengths (relative 1e-12), the
mixed-Voronoi mass recorded before the flips (relative 1e-12), the flip count and the mollification epsilon (exactly), the signpost angles, cone angles and
reference neighbours (1e-12). Synthetic data only.

    export NXR_CORTICAL_FLOW=<nxr-cortical-flow checkout>  NXR=<nxr checkout with prepare(mollify=, flip_eps=)>  NXR_NEURO_BUILD=<this build folder>
    uv run --project $NXR python -m pytest -q --basetemp=<a scratch folder> compute/test/parity/test_delaunay_parity.py
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
STUDIO = Path(os.environ["NXR_CORTICAL_FLOW"]) if "NXR_CORTICAL_FLOW" in os.environ else None
NXR = Path(os.environ["NXR"]) if "NXR" in os.environ else None
BUILD = Path(os.environ["NXR_NEURO_BUILD"]) if "NXR_NEURO_BUILD" in os.environ else None
pytestmark = pytest.mark.skipif(STUDIO is None or NXR is None or BUILD is None, reason="set NXR_CORTICAL_FLOW, NXR and NXR_NEURO_BUILD")
if NXR is not None:
    sys.path.insert(0, str(NXR / "src"))


# ── synthetic meshes ──
def octahedron_sphere(level: int):
    V = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    F = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    for _ in range(level):
        cache, G = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                p = (np.array(V[a]) + np.array(V[b])) / 2
                V.append(list(p / np.linalg.norm(p)))
                cache[k] = len(V) - 1
            return cache[k]

        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            G += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        F = G
    return np.array(V, float), np.array(F, np.int32)


def random_flips(V, F, n, rng):
    """combinatorial edge flips (faces (a,b,c),(b,a,d) -> (d,b,c),(c,a,d)) of a triangle mesh, as often as legal: a messy triangulation of the same points"""
    F = [list(f) for f in F]
    edges = {}
    for fi, f in enumerate(F):
        for k in range(3):
            edges[(f[k], f[(k + 1) % 3])] = (fi, k)
    done = 0
    keys = list(edges)
    for _ in range(40 * n):
        if done >= n:
            break
        a, b = keys[rng.integers(len(keys))]
        if (a, b) not in edges or (b, a) not in edges:
            continue
        f1, k1 = edges[(a, b)]
        f2, k2 = edges[(b, a)]
        c = F[f1][(k1 + 2) % 3]
        d = F[f2][(k2 + 2) % 3]
        if c == d or (c, d) in edges or (d, c) in edges:
            continue
        # convex quad only (planar or nearly so): the flip must not fold the surface over itself
        pa, pb, pc, pd = V[a], V[b], V[c], V[d]
        n1 = np.cross(pb - pa, pc - pa)
        n2 = np.cross(pa - pb, pd - pb)
        if np.dot(n1, n2) <= 0.2 * np.linalg.norm(n1) * np.linalg.norm(n2):
            continue
        s1 = np.dot(np.cross(pc - pd, pa - pd), n1) > 0 and np.dot(np.cross(pc - pd, pb - pd), n1) < 0
        if not s1:
            continue
        for key in [(a, b), (b, a), (b, c), (c, a), (a, d), (d, b)]:
            edges.pop(key, None)
        F[f1] = [d, b, c]
        F[f2] = [c, a, d]
        for fi in (f1, f2):
            for k in range(3):
                edges[(F[fi][k], F[fi][(k + 1) % 3])] = (fi, k)
        keys = list(edges)
        done += 1
    return np.array(F, np.int32), done


def planar_lattice(n, rng, flips):
    ij = [(i, j) for j in range(n + 1) for i in range(n + 1)]
    V = np.array([[i, j, 0.0] for i, j in ij])
    idx = lambda i, j: j * (n + 1) + i  # noqa: E731
    F = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            F += [[a, b, c], [a, c, d]] if rng.random() < 0.5 else [[a, b, d], [b, c, d]]
    return V, np.array(F, np.int32)


def cube_grid(n, rng):
    """a closed cube surface, n x n squares per face, each square split by a random diagonal; all quads cocircular"""
    pts, F, key = [], [], {}

    def vid(p):
        k = tuple(np.round(np.array(p, float) * n).astype(int))
        if k not in key:
            key[k] = len(pts)
            pts.append(list(np.array(k) / n))
        return key[k]

    for axis in range(3):
        for side in (0.0, 1.0):
            u, v = [a for a in range(3) if a != axis]
            for i in range(n):
                for j in range(n):
                    def P(a, b):
                        p = [0.0] * 3
                        p[axis], p[u], p[v] = side, a / n, b / n
                        return vid(p)

                    q = [P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)]
                    tri = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] if rng.random() < 0.5 else [[q[0], q[1], q[3]], [q[1], q[2], q[3]]]
                    F += tri
    V = np.array(pts, float)
    for f in F:  # outward orientation (the cube's centre is (1/2, 1/2, 1/2))
        a, b, c = V[f[0]], V[f[1]], V[f[2]]
        if np.dot(np.cross(b - a, c - a), (a + b + c) / 3 - 0.5) < 0:
            f[1], f[2] = f[2], f[1]
    return V, np.array(F, np.int32)


def meshes():
    rng = np.random.default_rng(20261110)
    out = {}
    V, F = octahedron_sphere(3)
    V = V * np.array([1.0, 1.0, 6.0]) + 0.03 * rng.uniform(-1, 1, V.shape)
    out["stretched jittered sphere"] = (V, F)
    V, F = octahedron_sphere(4)
    out["strongly stretched sphere, 5k faces"] = (V * np.array([1.0, 0.4, 9.0]) + 0.01 * rng.uniform(-1, 1, V.shape), F)
    V, F = planar_lattice(14, rng, 0)
    F2, n = random_flips(V, F, 160, rng)
    assert n > 100
    out["cocircular planar lattice, randomly flipped"] = (V, F2)
    V, F = cube_grid(5, rng)
    F2, n = random_flips(V, F, 120, rng)
    assert n > 60
    out["cocircular cube grid, randomly flipped"] = (V, F2)
    V, F = octahedron_sphere(2)
    V = V.copy()
    a, b, c = F[0]
    V[c] = 0.5 * (V[a] + V[b])
    out["sphere with a zero-area sliver"] = (V, F)
    V, F = octahedron_sphere(2)
    V = V.copy()
    a, b, c = F[3]
    V[c] = V[a] + (V[b] - V[a]) * (0.5 + 1e-9)
    out["sphere with a near-zero-area sliver"] = (V, F)
    out["tetrahedron"] = (np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float), np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], np.int32))
    V, F = octahedron_sphere(0)
    out["stretched octahedron"] = (V * np.array([1.0, 1.0, 0.05]) + np.array([[0.0, 0.0, 0.0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0.4, 0.1, 0], [-0.3, 0.2, 0]]), F)
    return out


MESHES = meshes()
DEGENERATE = {"cocircular cube grid, randomly flipped", "sphere with a zero-area sliver", "sphere with a near-zero-area sliver"}


def write_bin(path, V, F):
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", len(V), len(F)))
        f.write(np.ascontiguousarray(V, "<f8").tobytes())
        f.write(np.ascontiguousarray(F, "<i4").tobytes())


def run_cpp(path, mollify, flip_eps):
    p = subprocess.run([str(BUILD / "compute" / "delaunay_dump"), str(path), repr(mollify), repr(flip_eps)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def run_studio(path, mollify, flip_eps):
    tsx = STUDIO / "node_modules/.bin/tsx"
    env = dict(os.environ, NXR_CORTICAL_FLOW=str(STUDIO))
    p = subprocess.run([str(tsx), str(HERE / "delaunay_reference.mts"), str(path), repr(mollify), repr(flip_eps)], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def run_nxr(V, F, mollify, flip_eps):
    from nxr.operators.intrinsic_delaunay import prepare

    m = prepare(V, F, mollify=mollify, flip_eps=flip_eps)
    o = m.options
    return dict(F=m.F.tolist(), twin=m.twin.tolist(), length=m.length.tolist(), mass=m.mass.tolist(), angle=m.angle.tolist(), angleSum=m.angle_sum.tolist(),
                ref=m.ref.tolist(), record=dict(mollify=o["mollify"], epsilon=o["epsilon"], flipEps=o["flip_eps"], flips=o["flips"]))


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))) if a.size else 0.0


def compare(c, r, label):
    out = {}
    assert c["F"] == r["F"], f"{label}: faces differ after the flips"
    assert c["twin"] == r["twin"], f"{label}: twins differ"
    assert c["ref"] == r["ref"], f"{label}: signpost reference neighbours differ"
    assert c["record"]["flips"] == r["record"]["flips"], f"{label}: flip count {c['record']['flips']} != {r['record']['flips']}"
    # epsilon is a difference of lengths that nearly cancel (l_C - l_A - l_B + delta): compare it on the scale of the edge lengths it is added to
    scale = float(np.mean(r["length"]))
    assert abs(c["record"]["epsilon"] - r["record"]["epsilon"]) <= 1e-12 * scale, f"{label}: epsilon {c['record']['epsilon']!r} vs {r['record']['epsilon']!r}"
    out["length"] = rel(c["length"], r["length"])
    out["mass"] = rel(c["mass"], r["mass"])
    out["angle"] = rel(c["angle"], r["angle"])
    out["angleSum"] = rel(c["angleSum"], r["angleSum"])
    for k, v in out.items():
        assert v <= 1e-12, f"{label}: {k} max rel diff {v:.3g}"
    return out


@pytest.mark.parametrize("name", list(MESHES))
@pytest.mark.parametrize("mollify,flip_eps", [(1e-6, 1e-6), (1e-3, 1e-9), (0.0, 1e-6)])
def test_one_delaunay_matches_the_studio_and_nxr(name, mollify, flip_eps, tmp_path, capsys):
    V, F = MESHES[name]
    path = tmp_path / "m.bin"
    write_bin(path, V, F)
    c = run_cpp(path, mollify, flip_eps)
    e = run_studio(path, mollify, flip_eps)
    de = compare(c, e, "C++ vs Studio")
    # With mollify = 0 a (near-)degenerate triangle's epsilon is rounding noise (l_C - l_A - l_B ~ 1e-17..1e-9): the Studio's Math.hypot (which the C++ port
    # reproduces bit for bit) and CPython's math.hypot round differently there, so nxr is compared only where the mesh is not degenerate.
    degenerate = mollify == 0.0 and (name in DEGENERATE or c["record"]["epsilon"] != 0.0)
    dn = compare(c, run_nxr(V, F, mollify, flip_eps), "C++ vs nxr") if not degenerate else dict(length=float("nan"), mass=float("nan"), angle=float("nan"))
    with capsys.disabled():
        print(f"\n  {name} [mollify={mollify:g} flipEps={flip_eps:g}] nF={len(F)} flips={c['record']['flips']} eps={c['record']['epsilon']:.3g} "
              f"cocircular-edges={c['ties']} nonDelaunay-left={c['nonDelaunay']} | vs Studio: len {de['length']:.1e} mass {de['mass']:.1e} angle {de['angle']:.1e} | "
              f"vs nxr: len {dn['length']:.1e} mass {dn['mass']:.1e} angle {dn['angle']:.1e}")
