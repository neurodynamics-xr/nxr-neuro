"""Writes cores_golden.txt: the reference of nxr::select::corePatches, nxr's `_core_patches` (nxr/detect/vortex.py), on a SYNTHETIC unit-icosphere cortex (radius 80 mm, 2562 vertices) with PLANTED BLOBS: in each of
8 frames a few Gaussian bumps (both signs, different widths, some touching, some across the frame's other blobs) plus a little noise, as a hatted curl would show. The map is rounded to float32 first (nxr's
float32 maps; the same input reaches both). The threshold is the 98th percentile of |map| over all frames (the exact one: nothing sits within rounding of it). Run:
  uv run --project <nxr checkout> python compute/test/golden/gen_cores_golden.py compute/test/golden/cores_golden.txt
Written: the mesh (positions in mm, faces), the map [V x frames], the threshold, and nxr's cores in its own order (frame, sense, centre x y z in mm, area in mm^2); the C++ test derives the
neighbours from the faces. The attributes nxr-neuro adds (centre vertex, peak) are written too, computed here from the same patches (the vertex of the patch nearest the centre; the largest sense*value)."""
import sys

import numpy as np
import scipy.sparse as sp
from nxr.detect.vortex import _core_patches


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(level):
        cache, verts, faces = {}, list(V), []
        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = (verts[a] + verts[b]) / 2; verts.append(m / np.linalg.norm(m)); cache[k] = len(verts) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V, F = np.array(verts), np.array(faces)
    return V, F


U, F = icosphere(4)
R = 80.0
Vmm = U * R
nv, nf = len(U), len(F)
rng = np.random.default_rng(11)
# vertex areas: a third of each face's area to its vertices (barycentric)
a, b, c = Vmm[F[:, 0]], Vmm[F[:, 1]], Vmm[F[:, 2]]
fa = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
area = np.zeros(nv)
for k in range(3):
    np.add.at(area, F[:, k], fa / 3)
E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
A = sp.csr_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(nv, nv))
nfr = 8
X = 0.05 * rng.standard_normal((nv, nfr))
for f in range(nfr):
    for _ in range(int(rng.integers(2, 6))):
        ctr = U[rng.integers(nv)]
        width = rng.uniform(8, 22)                                     # mm
        amp = rng.uniform(0.8, 2.0) * (1 if rng.random() < 0.5 else -1)
        d = np.linalg.norm(Vmm - ctr * R, axis=1)
        X[:, f] += amp * np.exp(-0.5 * (d / width) ** 2)
X = X.astype(np.float32)
thr = float(np.percentile(np.abs(X.astype(np.float64)), 98))
fr, sn, cen, ar = _core_patches(X, thr, A.tocsr(), area, Vmm)
# the attributes nxr-neuro adds, from the same patches: re-find each core's vertices as the connected above-threshold set holding its centre's nearest vertex
from scipy.sparse.csgraph import connected_components
cv, pk = [], []
for k in range(len(fr)):
    f, sg = int(fr[k]), int(sn[k])
    m = sg * X[:, f].astype(np.float64) >= thr
    idx = np.flatnonzero(m)
    sub = A[idx][:, idx]
    nc, lab = connected_components(sub, directed=False)
    best = None
    # the patch whose area-weighted centre is this core's: match by centre
    for l in range(nc):
        mem = idx[lab == l]
        w = area[mem] * np.abs(X[mem, f])
        ce = (w[:, None] * Vmm[mem]).sum(0) / w.sum()
        if np.allclose(ce, cen[k], atol=1e-9) and abs(area[mem].sum() - ar[k]) < 1e-9:
            d = np.linalg.norm(Vmm[mem] - ce, axis=1)
            cv.append(int(mem[np.argmin(d)])); pk.append(float((sg * X[mem, f]).max())); best = l
            break
    assert best is not None
with open(sys.argv[1], "w") as out:
    out.write(f"nv {nv} nf {nf} frames {nfr} cores {len(fr)}\n")
    out.write("positions " + " ".join(repr(float(v)) for v in Vmm.ravel()) + "\n")
    out.write("faces " + " ".join(str(int(v)) for v in F.ravel()) + "\n")
    out.write("area " + " ".join(repr(float(v)) for v in area) + "\n")
    out.write("map " + " ".join(repr(float(v)) for v in X.ravel()) + "\n")                    # [V x frames] row-major
    out.write(f"threshold {thr!r}\n")
    out.write("frame " + " ".join(str(int(v)) for v in fr) + "\n")
    out.write("sense " + " ".join(str(int(v)) for v in sn) + "\n")
    out.write("centre " + " ".join(repr(float(v)) for v in cen.ravel()) + "\n")
    out.write("core_area " + " ".join(repr(float(v)) for v in ar) + "\n")
    out.write("centre_vertex " + " ".join(str(v) for v in cv) + "\n")
    out.write("peak " + " ".join(repr(v) for v in pk) + "\n")
print(f"wrote {sys.argv[1]}: {len(fr)} cores in {nfr} frames (threshold {thr:.3g}); senses {np.bincount((np.array(sn) > 0).astype(int))}")
