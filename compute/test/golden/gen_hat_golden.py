"""Writes hat_golden.txt: the Mexican hat gain of a spatial level, the reference of nxr::signal::graphfilterbank::mexicanHat and of apply_kernel's `spatial_shape: mexhat`.
The lines of nxr's `nxr/detect/vortex_cores.py hat_operator` that make the gain (the operator itself needs a cortex), on a SYNTHETIC sphere: hemisphere area A (mm²),
the Laplacian's spectrum l(l+1)/R² (l = 1 … 40, a sphere has no cortex), for ladder levels 1 … 5:
    lam_peak = 4π · 2^(level − 0.5) / A ;  s2 = 2 / lam_peak ;  g = lam · s2 · exp(−lam · s2 / 2) ;  g /= g.max()
(the geometric centre of the level's Weyl band [4π·2^(level−1)/A, 4π·2^level/A]). Run:
  uv run --project <nxr checkout> python compute/test/golden/gen_hat_golden.py compute/test/golden/hat_golden.txt"""
import sys

import numpy as np

R = 80.0                                   # mm
A = 4 * np.pi * R ** 2 / 2                 # a hemisphere's area (mm²): the sphere's half
lam = np.array([l * (l + 1) / R ** 2 for l in range(1, 41)])
out = open(sys.argv[1], "w")
out.write(f"area {A!r}\n")
out.write("lam 40 " + " ".join(repr(float(v)) for v in lam) + "\n")
for level in range(1, 6):
    lam_peak = 4 * np.pi * 2 ** (level - 0.5) / A
    s2 = 2 / lam_peak
    g = lam * s2 * np.exp(-lam * s2 / 2)
    gmax = g.max()                          # nxr scales by the SAMPLE maximum (a cortex's dense spectrum: the hat's own 2/e to the spectrum's resolution); the sphere's coarse one falls short of it
    out.write(f"peak {level} {lam_peak!r} {float(gmax)!r}\n")
    out.write(f"raw {level} 40 " + " ".join(repr(float(v)) for v in g) + "\n")
    out.write(f"gain {level} 40 " + " ".join(repr(float(v)) for v in g / gmax) + "\n")
out.close()
print(f"wrote {sys.argv[1]}")
