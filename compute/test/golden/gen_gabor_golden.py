"""Writes gabor_golden.txt: the reference of the Gabor bursts' compute functions (nxr::signal::timefilterbank::gaborGrid, nxr::signal::gaborFit and gaborWidthPeak, nxr::select::suppress
and tangentEdges) on a SYNTHETIC signal: the reference detector's grid (`widths`, `channel_atoms`), its candidates (`peaks`), each candidate's fit (`refine`) and burst test (`is_burst`), the merge
of the bursts (`merge`), and each merged burst's tangent onset and offset (`tangent_edges`). The candidates are the reference's own (on its decimated grid): the C++ test fits the same ones.
The reference is the Python detector the spec ports (docs/specs/2026-11-20-gabor-bursts.md): the folder of its modules `gabor` and `spacetime` is given by NXR_GABOR_REFERENCE.
Run:  NXR_GABOR_REFERENCE=<folder> uv run --project <nxr checkout> python compute/test/golden/gen_gabor_golden.py compute/test/golden/gabor_golden.txt"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ["NXR_GABOR_REFERENCE"])
import gabor as g  # noqa: E402
import spacetime as st  # noqa: E402

fs, T = 250.0, 5000                                                     # 20 s
t = np.arange(T) / fs
rng = np.random.default_rng(4)
x = 0.02 * rng.standard_normal(T)
for t0, f, s, A, phi in ((5.0, 12.0, 0.15, 1.0, 0.7), (10.0, 25.0, 0.06, 0.8, -1.0), (14.0, 18.0, 0.30, 0.9, 2.0)):
    x += A * np.exp(-(t - t0) ** 2 / (2 * s * s)) * np.cos(2 * np.pi * f * (t - t0) + phi)
fmin, fmax, smin, smax, per_oct, mincyc = 8.0, 35.0, 0.025, 0.64, 4, 2.0
sig = g.widths(smin, smax, per_oct)
grid = [(s, fr) for s, fr, _, _ in g.channel_atoms(x, fs, sig, fmin, fmax)]
P = g.peaks(g.channel_atoms(x, fs, sig, fmin, fmax), 4.0)
R = [g.refine(x, fs, p) for p in P]
ok = [fmin <= r["f"] <= fmax and g.is_burst(x, fs, r, sig, mincyc) for r in R]
passing = [r for r, o in zip(R, ok) if o]
merged = g.merge(passing)
order = [next(i for i, q in enumerate(passing) if q is m) for m in merged]
ev = [dict(m, burst=True, support=3, centre=0) for m in merged]
st.tangent_edges(x[None, :], fs, ev)
r = lambda v: repr(float(v))
with open(sys.argv[1], "w") as out:
    out.write(f"rate {r(fs)} band {r(fmin)} {r(fmax)} widths {r(smin)} {r(smax)} {per_oct} min_f_sigma 0.5 mincyc {r(mincyc)}\n")
    out.write(f"x 1 {T}\n" + " ".join(r(v) for v in x) + "\n")
    out.write(f"sig 1 {len(sig)}\n" + " ".join(r(v) for v in sig) + "\n")
    out.write(f"grid {len(grid)}\n")
    for s, fr in grid:
        out.write(f"width {r(s)} freqs 1 {len(fr)}\n" + " ".join(r(v) for v in fr) + "\n")
    out.write(f"candidates {len(P)}\n")
    for p, q, o in zip(P, R, ok):
        out.write(f"candidate {r(p['t'])} {r(p['f'])} {r(p['sigma'])} fit {r(q['f'])} {r(q['sigma'])} {r(q['amp'])} {r(q['phase'])} {r(q['W'])} {r(q['cycles'])} burst {int(o)}\n")
    out.write(f"merged 1 {len(order)}\n" + " ".join(str(i) for i in order) + "\n")
    for e in ev:
        out.write(f"edges {r(e.get('tan_on', float('nan')))} {r(e.get('tan_off', float('nan')))}\n")
print(len(sig), "widths,", sum(len(fr) for _, fr in grid), "atoms,", len(P), "candidates,", len(passing), "bursts,", len(merged), "merged;",
      [(round(m["t"], 3), round(m["f"], 2), round(m["sigma"], 3), round(m["amp"], 3)) for m in merged])
