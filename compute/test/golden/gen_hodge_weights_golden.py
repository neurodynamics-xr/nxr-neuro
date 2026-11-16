"""golden for test_hodge_weights: nxr.inverse.fused (glued_kernel, fused_weights, synthesis) on a tiny synthetic glued icosphere.
Run: cd <nxr checkout> && uv run python <this file> <out.txt>"""
import sys
import numpy as np
from scipy.spatial import ConvexHull
from nxr.register.modes import Solve
from nxr.inverse.fused import glued_kernel, fused_weights, synthesis, FaceOperators

rng = np.random.default_rng(7)
# icosphere level 1 (42 vertices), from the convex hull of the unit icosahedron + edge midpoints
t = (1 + 5 ** .5) / 2
V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
V /= np.linalg.norm(V, axis=1, keepdims=True)
E = set()
H = ConvexHull(V)
for s in H.simplices:
    for i in range(3):
        E.add(tuple(sorted((s[i], s[(i + 1) % 3]))))
mid = np.array([V[a] + V[b] for a, b in sorted(E)])
V = np.vstack([V, mid / np.linalg.norm(mid, axis=1, keepdims=True)])
F = ConvexHull(V).simplices.copy()
for i, f in enumerate(F):
    if np.dot(np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]]), V[f].mean(0)) < 0:
        F[i] = f[[0, 2, 1]]
V = (V * 0.07).astype(np.float32).astype(np.float64)       # a head-sized sphere in metres
nV = len(V)
a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
mass = np.zeros(nV)
for k in range(3):
    np.add.at(mass, F[:, k], area / 3)
# smooth potentials: random combinations of the monomials of degree <= 3
x = V / 0.07
mon = np.stack([x[:, 0] ** i * x[:, 1] ** j * x[:, 2] ** k for i in range(4) for j in range(4) for k in range(4) if 0 < i + j + k <= 3], 0)
K = 12
P = (rng.standard_normal((K, mon.shape[0])) @ mon).astype(np.float32)
kind = np.array([0, 2] * (K // 2))
mu = np.linspace(10, 60, K)
nb = 4                                                     # the last 4 vertices are a bridge; the input holds 5 cut wall vertices instead
kept = nV - nb
perm = rng.permutation(kept + 5)
keep = np.r_[perm[:kept], -np.ones(nb, int)]
cutv = perm[kept:]
Vin = np.zeros((kept + 5, 3))
Vin[keep[:kept]] = V[:kept]
Vin[cutv] = V[kept:].mean(0) + 0.01 * rng.standard_normal((5, 3)) + np.array([0, 0, 0.03])
Vin = Vin.astype(np.float32).astype(np.float64)
solve = Solve(P, mu, kind, keep, mass, V, "m", "s")
C = 20
W = rng.standard_normal((3 * len(Vin), C)) * 1e-8
Wg = glued_kernel(W, solve, Vin)
B = fused_weights(solve, F, Wg)
Xc = rng.standard_normal((C, 3))
J = synthesis(solve, F, B @ Xc)
with open(sys.argv[1], "w") as o:
    def w(name, M):
        M = np.atleast_2d(np.asarray(M, float))
        o.write(f"{name} {M.shape[0]} {M.shape[1]}\n" + " ".join(f"{v:.17g}" for v in M.ravel()) + "\n")
    w("positions", V); w("faces", F); w("P", P.astype(np.float64)); w("mu", mu); w("kind", kind); w("mass", mass); w("keep", keep); w("input_positions", Vin)
    w("W", W); w("Wg", Wg); w("B", B); w("X", Xc); w("J", J)
