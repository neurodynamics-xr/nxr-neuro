"""Writes rollup_golden.txt: the reference of nxr::io::rollup (the measurement pyramid's read-time roll-up) — nxr's `nxr.measure.rollup` (`space`, `time`) on random arrays, every merge (sum, max, min), every level, several
tower anchors (`code0`), and a DERIVED density (Σ power / (node area × tile samples), the Studio's `searchAtlas` normalisation: the sums rolled up first, then the ratio).
Run:  uv run --project <nxr checkout> python io/test/golden/gen_rollup_golden.py io/test/golden/rollup_golden.txt"""
import sys

import numpy as np
from nxr.measure import rollup

R = lambda v: repr(float(v))
out = open(sys.argv[1], "w")
rng = np.random.default_rng(3)
depth, rows = 4, 3
leaves = 2 * 2 ** depth
for op in ("sum", "max", "min"):
    x = rng.standard_normal((rows, leaves)) * 5
    out.write(f"space {op} {depth} {rows} {leaves}\nx " + " ".join(R(v) for v in x.ravel()) + "\n")
    for level in range(-1, depth + 1):
        y = rollup.space(x, depth, level, op, axis=1)
        out.write(f"level {level} {y.shape[1]}\ny " + " ".join(R(v) for v in y.ravel()) + "\n")
frames = 150
for op in ("sum", "max", "min"):
    for code0 in (0, 5, 77, 1023, 123456):
        x = rng.standard_normal((rows, frames)) * 3
        out.write(f"time {op} {code0} {rows} {frames}\nx " + " ".join(R(v) for v in x.ravel()) + "\n")
        for m in (0, 1, 3, 4, 8):
            y = rollup.time(x, m, op, axis=1, code0=code0)
            out.write(f"level {m} {y.shape[1]}\ny " + " ".join(R(v) for v in y.ravel()) + "\n")
# a derived density, both orders of roll-up
code0 = 4321
power = np.abs(rng.standard_normal((leaves, frames))) + 0.1
area = np.abs(rng.standard_normal(leaves)) + 0.5
samples = rng.integers(0, 80, frames).astype(float)
out.write(f"density {depth} {leaves} {frames} {code0}\npower " + " ".join(R(v) for v in power.ravel()) + "\narea " + " ".join(R(v) for v in area) + "\nsamples " + " ".join(R(v) for v in samples) + "\n")
for level, m in ((depth, 0), (2, 3), (-1, 6), (0, 1), (3, 8)):
    p = rollup.time(rollup.space(power, depth, level, "sum", axis=0), m, "sum", axis=1, code0=code0)
    p2 = rollup.space(rollup.time(power, m, "sum", axis=1, code0=code0), depth, level, "sum", axis=0)
    assert np.allclose(p, p2)                       # the roll-ups commute
    a = rollup.space(area, depth, level, "sum")
    s = rollup.time(samples, m, "sum", code0=code0)
    d = p / (a[:, None] * s[None, :])
    d[~np.isfinite(d)] = np.nan
    out.write(f"dlevel {level} {m} {p.shape[0]} {p.shape[1]}\nd " + " ".join("nan" if np.isnan(v) else R(v) for v in d.ravel()) + "\n")
out.close()
