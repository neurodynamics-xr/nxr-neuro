"""Writes magnetic_golden.txt: the reference for nxr::operators::hermitianEigs. A magnetic Laplacian L = D - W on a random graph (400 vertices, about 1600
edges, random strengths and lags; seeded), its lowest 8 eigenpairs by scipy's eigsh (ARPACK's complex Hermitian solver, shift-invert at -1e-6). Synthetic
data only.
Format: `n m k`, then m lines `u v re(W_uv) im(W_uv)`, then k eigenvalues, then the k eigenvectors, one per line (re im pairs, n each).
Run:  uv run --project <nxr checkout> python compute/test/golden/gen_magnetic_golden.py compute/test/golden/magnetic_golden.txt"""
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

rng = np.random.default_rng(20261112)
n, k = 400, 8
edges = set()
for u in range(n):                                   # a ring (connected) plus random chords
    edges.add((u, (u + 1) % n))
while len(edges) < 1600:
    u, v = rng.integers(0, n, 2)
    if u != v and (u, v) not in edges and (v, u) not in edges:
        edges.add((int(u), int(v)))
edges = sorted(edges)
w = rng.uniform(0.2, 1.0, len(edges)) * np.exp(1j * rng.uniform(-np.pi, np.pi, len(edges)))
rows = [u for u, v in edges] + [v for u, v in edges]
cols = [v for u, v in edges] + [u for u, v in edges]
W = sp.csr_matrix((np.concatenate([w, np.conj(w)]), (rows, cols)), shape=(n, n))
D = sp.diags(np.asarray(abs(W).sum(axis=1)).ravel())
L = (D - W).tocsc()
vals, vecs = eigsh(L, k=k, sigma=-1e-6, which="LM", tol=1e-12)
order = np.argsort(vals)
vals, vecs = vals[order], vecs[:, order]
with open(sys.argv[1], "w") as f:
    f.write(f"{n} {len(edges)} {k}\n")
    for (u, v), x in zip(edges, w):
        f.write(f"{u} {v} {x.real:.17g} {x.imag:.17g}\n")
    f.write(" ".join(f"{x:.17g}" for x in vals) + "\n")
    for j in range(k):
        f.write(" ".join(f"{z.real:.17g} {z.imag:.17g}" for z in vecs[:, j]) + "\n")
