"""Writes repeat_golden.txt: the reference of nxr::select::track_crossings::repeat, nxr's `repeat` and `tile_dot` (nxr/detect/vortex_cores.py), on PLANTED TILE SUMS: 24 tiles, two senses each (nxr's tile-sense index 2 t + (sense > 0)), two runs whose tiles'
mean resultants share a direction (with noise) in most tiles, oppose it in some, and have too few crossings in a few (below nxr's 10). The sums are 3-D tangent vectors at a tile's centre and the counts are the crossings, as
`Tiles.crossings` returns them. Run:
  uv run --project <nxr checkout> python compute/test/golden/gen_repeat_golden.py compute/test/golden/repeat_golden.txt
Written: both runs' sums [2T x 3] and counts [2T], nxr's per-tile dot (NaN written as nan) over the senses pooled, and its count-weighted mean cosine."""
import sys

import numpy as np

import nxr.detect.vortex_cores as vc

rng = np.random.default_rng(31)
T = 24
dirs = rng.standard_normal((T, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)


def run(sign, noise, few):
    sums = np.zeros((2 * T, 3)); n = np.zeros(2 * T, int)
    for t in range(T):
        for s in range(2):
            k = int(rng.integers(2, 12)) if t in few else int(rng.integers(6, 30))
            u = sign[t] * dirs[t] + noise * rng.standard_normal(3)
            sums[2 * t + s] = k * 0.8 * u / max(np.linalg.norm(u), 1e-9) * rng.uniform(0.4, 1.0); n[2 * t + s] = k
    return sums, n


sign = np.ones(T); sign[[3, 8, 15]] = -1                       # three tiles go the opposite way in run B
a = run(np.ones(T), 0.35, few={5, 11})
b = run(sign, 0.35, few={11, 19})
d = vc.tile_dot(a, b); r = vc.repeat(a, b)
with open(sys.argv[1], "w") as f:
    f.write(f"tiles {T} min {vc.NMIN_TILE}\n")
    for nm, (s, n) in (("a", a), ("b", b)):
        f.write(f"sums_{nm} " + " ".join(repr(float(v)) for v in s.ravel()) + "\n")
        f.write(f"n_{nm} " + " ".join(str(int(v)) for v in n) + "\n")
    f.write("dot " + " ".join(repr(float(v)) for v in d) + "\n")
    f.write(f"cosine {float(r)!r}\n")
print(f"wrote {sys.argv[1]}: cosine {r:.4f}, {int(np.isfinite(d).sum())} of {T} tiles compared, dots {np.nanmin(d):.2f} .. {np.nanmax(d):.2f}")
