"""Golden for nxr::stats::NumpyGenerator and signFlipConsistency: numpy's default_rng (SeedSequence + PCG64) and the sign-flip test of nxr's flow maps
(nxr `nxr.flow.flowmap.flip_test`, copied below unchanged in its computation) on synthetic per-cell vector fields, written to sign_flip_golden.txt.
Run:  uv run --project <nxr checkout> python gen_sign_flip_golden.py sign_flip_golden.txt"""
import sys
import numpy as np


def flip_test(Pseg, perms, rng):
    S, n = Pseg.shape[:2]; U = (Pseg / np.maximum(np.linalg.norm(Pseg, axis=2, keepdims=True), 1e-30)).reshape(S, -1)
    R = np.linalg.norm(U.sum(0).reshape(n, 3), axis=1) / S; E = np.where(rng.random((perms, S)) < .5, -1., 1.); ge = np.zeros(n); mx, md = [], []
    for c0 in range(0, perms, 50):
        Rn = np.linalg.norm((E[c0:c0 + 50] @ U).reshape(-1, n, 3), axis=2) / S; ge += (Rn >= R).sum(0); mx += Rn.max(1).tolist(); md += np.median(Rn, 1).tolist()
    mx, md = np.array(mx), np.array(md)
    return dict(R1=R, p=(ge + 1) / (perms + 1), pfwe=(np.sum(mx[:, None] >= R[None], 0) + 1) / (perms + 1), median=float(np.median(R)),
                median_null95=float(np.percentile(md, 95)), median_p=float((np.sum(md >= np.median(R)) + 1) / (perms + 1)), fwe_thr=float(np.percentile(mx, 95)))


out = open(sys.argv[1], "w")
out.write("# sign-flip golden: numpy default_rng draws and nxr's flip_test on synthetic cells; see gen_sign_flip_golden.py\n")
for seed in (0, 12345, 2 ** 40 + 7):
    g = np.random.default_rng(seed)
    g = np.random.default_rng(seed)
    raw = g.bit_generator.random_raw(4)
    g = np.random.default_rng(seed)
    d = g.random(6)
    out.write(f"seed {seed}\nraw {' '.join(str(int(v)) for v in raw)}\nrandom {' '.join(repr(float(v)) for v in d)}\n")
data = np.random.default_rng(99)
cells, elements, perms = 23, 17, 120
P = data.standard_normal((cells, elements, 3))
P[:, :6, 0] += 1.5                       # six elements with a consistent direction
P[3, 2] = 0.0                            # a zero vector: its unit vector is 0
P = np.array([float(f"{v:.12g}") for v in P.ravel()]).reshape(P.shape)
r = flip_test(P, perms, np.random.default_rng(0))
out.write(f"cells {cells} elements {elements} perms {perms}\n")
out.write("P " + " ".join(repr(float(v)) for v in P.ravel()) + "\n")
for k in ("R1", "p", "pfwe"):
    out.write(k + " " + " ".join(repr(float(v)) for v in r[k]) + "\n")
for k in ("median", "median_null95", "median_p", "fwe_thr"):
    out.write(f"{k} {r[k]!r}\n")
