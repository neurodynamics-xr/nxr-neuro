"""Writes tower_golden.txt: the reference of nxr::tower (codes, placements, the frame tiling's count and origin, the time roll-up on the tower) — nxr's D135 TOWER, `pipelines/atlas/tower.py`
(`Placement.sample_codes`, `period_s`, `level_for_seconds`) and `measure/rollup.py` `time(..., code0=)`, on SYNTHETIC placements and arrays.
Run:  uv run --project <nxr checkout> python compute/test/golden/gen_tower_golden.py compute/test/golden/tower_golden.txt"""
import sys

import numpy as np
from nxr.measure.rollup import time as roll_time
from nxr.pipelines.atlas.tower import Placement, level_for_seconds, period_s

rng = np.random.default_rng(5)
out = open(sys.argv[1], "w")
R = lambda v: repr(float(v))

# placements: (name, fs, n, start_s, day, levels)
cases = [("recording", 1200.0, 4000, None, None, [-18, -20, -10, 0, 3]),
         ("day", 250.0, 3000, 43200.123456, None, [-18, -22, -25, 0]),
         ("day_late", 300.0, 2500, 86399.9, 11, [-18, -12, 0, 2]),
         ("birth", 600.0, 1800, 3600.5, 7000, [-18, -21])]
for name, fs, n, s0, day, levels in cases:
    p = Placement(name, fs, n, s0, day)
    out.write(f"placement {name} {R(fs)} {n} {'nan' if s0 is None else R(s0)} {-1 if day is None else day} {p.anchor} {'none' if p.coarsest is None else p.coarsest}\n")
    for L in levels:
        try:
            c = p.sample_codes(L)
        except ValueError:
            out.write(f"refused {L}\n")
            continue
        out.write(f"codes {L} {R(period_s(L))} {c[0]} {c[-1]} {c[-1] - c[0] + 1}\n")
        out.write("samples " + " ".join(str(int(v)) for v in c[::7]) + "\n")      # every 7th sample's code
        # per-frame sample counts (the atlas's `samples` count, nxr scalars: bincount of the codes)
        cnt = np.bincount(c - c[0])
        out.write("counts " + " ".join(str(int(v)) for v in cnt[:400]) + "\n")
for s in (0.001, 0.0025, 1 / 360, 1 / 432, 1.0, 86400.0, 100000.0, 3600.0):
    out.write(f"level_for_seconds {s!r} {level_for_seconds(s)}\n")

# the roll-up on the tower
for op in ("sum", "max", "min"):
    for m, code0 in ((1, 0), (2, 0), (2, 5), (3, 13), (3, 16), (4, 7)):
        x = rng.standard_normal((3, 37)).astype(np.float64)
        y = roll_time(x, m, op, -1, code0)
        out.write(f"rollup {op} {m} {code0} {x.shape[0]} {x.shape[1]} {y.shape[1]}\n")
        out.write("x " + " ".join(R(v) for v in x.ravel()) + "\n")
        out.write("y " + " ".join(R(v) for v in y.ravel()) + "\n")
out.close()
