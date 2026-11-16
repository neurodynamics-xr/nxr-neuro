"""Writes histogram_golden.txt: nxr's `_core_threshold` (nxr/detect/vortex.py), the reference of nxr::histogram: the percentile of a non-negative series from
a 4096-bin log histogram whose range is set by the first chunk (hi = 4 x its maximum, lo = hi * 1e-6), values above hi clipped into the top bin. Synthetic
data only: 4 rows x 7000 frames of log-normal float32 values, with spikes far above hi in later chunks (the clipping), in chunks of 3000 frames; the
percentiles 50, 75, 98 (CORE_PCT set for each), and the counts of the reference's own loop (copied from `_core_threshold`, its threshold checked equal).
Run:  uv run --project <nxr checkout> python compute/test/golden/gen_histogram_golden.py compute/test/golden/histogram_golden.txt"""
import sys

import numpy as np

import nxr.detect.vortex as vx

rng = np.random.default_rng(20261113)
R, T, CHUNK = 4, 7000, 3000
X = rng.lognormal(0.0, 1.5, (R, T)).astype(np.float32)
X[1, 4000] = X[:, :CHUNK].max() * 50          # above hi = 4 x the first chunk's maximum: clipped into the top bin
X[2, 6500] = X[:, :CHUNK].max() * 9
X[0, 10:20] = 0                              # below lo: bin 0
Hc = np.eye(R, dtype=np.float32)
fr = np.arange(T)

# the reference's own loop (vortex._core_threshold), for its counts
nb = 4096; hi = lo = sc = None; hist = np.zeros(nb, np.int64)
for i in range(0, fr.size, CHUNK):
    a = np.abs(Hc @ X[:, fr[i:i + CHUNK]].astype(np.float32)).ravel()
    if hi is None: hi = float(a.max()) * 4; lo = hi * 1e-6; sc = (nb - 1) / np.log(hi / lo)
    a = np.minimum(a, hi); b = np.zeros(a.size, np.int64); m = a >= lo; b[m] = np.minimum(np.floor(np.log(a[m] / lo) * sc).astype(np.int64) + 1, nb - 1)
    hist += np.bincount(b, minlength=nb)

pct = {}
for p in (50.0, 75.0, 98.0):
    vx.CORE_PCT = p
    pct[p] = vx._core_threshold(Hc, X, fr, CHUNK)
    assert pct[p] == float(np.geomspace(lo, hi, nb)[np.searchsorted(np.cumsum(hist) / hist.sum(), p / 100)])

with open(sys.argv[1], "w") as f:
    f.write(f"data {R} {T} {CHUNK}\n")
    f.write(" ".join(f"{v:.9g}" for v in X.ravel()) + "\n")
    f.write(f"bins {nb} {lo!r} {hi!r}\n")
    for p, v in pct.items(): f.write(f"percentile {p!r} {v!r}\n")
    f.write("counts " + " ".join(str(int(c)) for c in hist) + "\n")
