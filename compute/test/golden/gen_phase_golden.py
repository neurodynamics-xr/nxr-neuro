"""Writes phase_golden.txt: nxr's phase reader (nxr/detect/core_phase.py: `gradient_and_kernel`'s GV, the P1 gradient onto the vertices, and `read_out`'s
formulas: grad phi = Im(conj(c) grad c) / max(|c|^2, 1e-30), projected on the vertex normals; u = -om grad phi / max(|grad phi|^2, 1e-12)), the reference of
nxr::signal, on a synthetic closed surface: an icosphere (level 3, radius 50 mm, bumped), area-weighted vertex normals, and a random complex scalar field
(3 frames). Synthetic data only.
Run:  uv run --project <nxr checkout> python compute/test/golden/gen_phase_golden.py compute/test/golden/phase_golden.txt"""
import sys

import numpy as np
from scipy.spatial import cKDTree

from nxr.detect.core_phase import gradient_and_kernel

rng = np.random.default_rng(20261112)


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6],
         [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    V = [list(np.array(v) / np.linalg.norm(v)) for v in V]
    for _ in range(level):
        mid, nF = {}, []

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = (np.array(V[a]) + np.array(V[b])) / 2
                V.append(list(p / np.linalg.norm(p)))
                mid[k] = len(V) - 1
            return mid[k]
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        F = nF
    V = np.array(V)
    r = 1 + 0.1 * np.sin(3 * V[:, 0]) * np.cos(2 * V[:, 2])
    return 50.0 * V * r[:, None], np.array(F, np.int64)


P, F = icosphere(3)
nv = len(P)
fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
fa = np.linalg.norm(fn, axis=1) / 2
N = np.zeros_like(P)
for s in range(3):
    np.add.at(N, F[:, s], fn)                       # area-weighted (|fn| = 2A) vertex normals
N /= np.linalg.norm(N, axis=1, keepdims=True)
ar = np.zeros(nv)
for s in range(3):
    np.add.at(ar, F[:, s], fa / 3)
GV, _ = gradient_and_kernel(P, ar, F, np.arange(nv), nv, 20.0, cKDTree(P))
c = rng.standard_normal((nv, 3)) + 1j * rng.standard_normal((nv, 3))          # [V x m] complex
gc = np.stack([G @ c for G in GV], -1)                                         # [V x m x 3]
rho2 = np.abs(c) ** 2
gp = np.imag(np.conj(c)[..., None] * gc) / np.maximum(rho2, 1e-30)[..., None]
gp -= (gp * N[:, None, :]).sum(-1, keepdims=True) * N[:, None, :]
om = 2 * np.pi * 10.0
u = -om * gp / np.maximum((gp ** 2).sum(-1, keepdims=True), 1e-12)

out = open(sys.argv[1], "w")


def line(tag, a):
    a = np.asarray(a).ravel()
    out.write(f"{tag} {a.size} " + " ".join(repr(float(v)) for v in a) + "\n")


out.write(f"mesh {nv} {len(F)}\n")
line("positions", P)
line("faces", F)
line("normals", N)
line("re", c.real)
line("im", c.imag)
line("grad_phi", gp)        # [V x m x 3]
line("velocity", u)         # [V x m x 3]
out.close()
print(f"wrote {sys.argv[1]}: {nv} vertices")
