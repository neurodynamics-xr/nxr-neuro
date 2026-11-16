"""Writes pyramid_golden.txt: the reference of nxr::pyramid (the measurement pyramid's band power on leaves × frames) — nxr's `nxr.measure.scalars` (`band_gains`, `analytic_bands`, `band_power`)
and `nxr.geom.trees.subject_tree` on a SYNTHETIC two-hemisphere icosphere mesh, a random kernel, random sensor data, a day-anchored tower placement (a partial first frame) and a bad span.
Run:  uv run --project <nxr checkout> python compute/test/golden/gen_pyramid_golden.py compute/test/golden/pyramid_golden.txt"""
import sys

import numpy as np
from nxr.geom.ladder import vertex_areas
from nxr.geom.trees import subject_tree
from nxr.measure import scalars, vectors
from nxr.pipelines.atlas.tower import Placement

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
    return np.array(v), np.array(f, dtype=np.int64)


rng = np.random.default_rng(11)
U, F0 = icosphere(2)
m = len(U)
P = np.vstack([U * [2.0, 1.0, 1.5] + [0, 1.3, 0], U * [2.0, 1.0, 1.5] + [0, -1.3, 0]]) * 0.05     # metres: a small brain
F = np.vstack([F0, F0 + m])
structures = np.r_[np.full(m, 1), np.full(m, 2)]
V = len(P)
area = vertex_areas(P, F)
depth = 3
tree = subject_tree(P, F, structures, None, depth)
finest = tree.finest
M, n, sfreq = 6, 1000, 250.0
K = rng.standard_normal((V, M)).astype(np.float32) * 1e-3
data = rng.standard_normal((M, n)).astype(np.float64)
data += 3 * np.sin(2 * np.pi * 10 * np.arange(n) / sfreq)[None, :] * rng.standard_normal((M, 1))
good = np.ones(n, bool)
good[300:361] = False
level = -18
pl = Placement("rec", sfreq, n, 43200.1234)
codes = pl.sample_codes(level)
bands = scalars.octave_bands()
res = scalars.band_power(tree, area, K, data, sfreq, bands, chunk_frames=4, good=good, codes=codes)
n_frames = int(codes[-1] - codes[0] + 1)

out.write(f"mesh {V} {len(F)} {m}\n")
out.write("P " + " ".join(R(v) for v in P.ravel()) + "\n")
out.write("F " + " ".join(str(int(v)) for v in F.ravel()) + "\n")
out.write(f"tree {depth} {finest}\n")
out.write("codes " + " ".join(str(int(v)) for v in tree.codes) + "\n")
out.write("area " + " ".join(R(v) for v in area) + "\n")
out.write(f"kernel {V} {M}\n")
out.write("K " + " ".join(R(v) for v in K.astype(np.float64).ravel()) + "\n")        # float32 values, widened
out.write(f"data {M} {n} {R(sfreq)}\n")
out.write("X " + " ".join(R(v) for v in data.ravel()) + "\n")
out.write("good " + " ".join("1" if g else "0" for g in good) + "\n")
out.write(f"placement {R(43200.1234)} {level} {int(codes[0])} {n_frames}\n")
out.write("sample_codes " + " ".join(str(int(c)) for c in codes) + "\n")
out.write(f"bands {len(bands)}\n" + "\n".join(f"band {R(lo)} {R(hi)}" for lo, hi in bands) + "\n")
# the power-partition bank's gains on a grid of frequencies
fr = np.linspace(0, 125, 77)
g = scalars.band_gains(fr, bands)
out.write(f"gains {len(fr)}\nfreqs " + " ".join(R(v) for v in fr) + "\n")
for b in range(len(bands)):
    out.write("gain " + " ".join(R(v) for v in g[b]) + "\n")
# the analytic signal of two bands of the first two channels (nxr: one FFT over the whole record, float32)
for b, A in scalars.analytic_bands(data, sfreq, bands):
    if b in (1, 4):
        out.write(f"analytic {b} 2 {n}\n")
        out.write("re " + " ".join(R(v) for v in A[:2].real.ravel()) + "\n")
        out.write("im " + " ".join(R(v) for v in A[:2].imag.ravel()) + "\n")
# the result: [band, leaf, frame]
out.write(f"result {len(bands)} {tree.n_leaves} {n_frames}\n")
for key in ("power", "env", "envmax"):
    out.write(key + " " + " ".join(R(v) for v in res[key].ravel()) + "\n")
out.write("w " + " ".join(R(v) for v in res["w"]) + "\n")
out.write("samples " + " ".join(str(int(v)) for v in res["samples"]) + "\n")
# THE FREE-ORIENTATION TENSOR (nxr `vectors.band_tensor`): synthetic canonical frames on the two ellipsoids (n outward from each hemisphere's centre, e1 = z projected on the tangent plane, e2 = n × e1;
# the vertices within 0.1 rad of a pole are singular: NaN), a free kernel [3V, M] and the same data, bad span and tower codes
cen = np.vstack([np.tile([0, 1.3, 0], (m, 1)), np.tile([0, -1.3, 0], (m, 1))]) * 0.05
nrm = P - cen
nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
z = np.array([0, 0, 1.0])
e1 = z - (nrm @ z)[:, None] * nrm
sing = np.linalg.norm(e1, axis=1) < 0.1
e1 = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
e2 = np.cross(nrm, e1)
sing[[5, 100, 200]] = True                      # a few more with no frame
for a in (e1, e2, nrm):
    a[sing] = np.nan
fr = vectors.Frames(e1, e2, nrm, sing)
K3 = (rng.standard_normal((3 * V, M)) * 1e-3).astype(np.float32)
tres = vectors.band_tensor(tree, area, fr, K3, data, sfreq, bands, chunk_frames=4, good=good, codes=codes)
out.write(f"frames {V} {int(sing.sum())}\n")
out.write("singular " + " ".join(str(int(i)) for i in np.flatnonzero(sing)) + "\n")
for tag, a in (("E1", e1), ("E2", e2), ("EN", nrm)):
    out.write(tag + " " + " ".join(R(v) for v in np.nan_to_num(a).ravel()) + "\n")
out.write("K3 " + " ".join(R(v) for v in K3.astype(np.float64).ravel()) + "\n")
out.write(f"tensor {len(bands)} {tree.n_leaves} {n_frames}\n")
for key in vectors.TENSOR_KEYS:
    out.write("T_" + key + " " + " ".join(R(v) for v in tres["tensor"][key].ravel()) + "\n")
out.write("T_samples " + " ".join(str(int(v)) for v in tres["samples"]) + "\n")
out.close()
