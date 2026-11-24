"""Golden for nxr::signal::fitAperiodic: nxr's `nxr.spectral.aperiodic.fit` (the lab's specparam port) on synthetic spectra (a 1/f background with a knee, two
peaks, multiplicative noise), without and with the knee, written to aperiodic_golden.txt.
Run:  uv run --project <nxr checkout> python gen_aperiodic_golden.py aperiodic_golden.txt"""
import sys
import numpy as np
from nxr.spectral import aperiodic

rng = np.random.default_rng(5)
f = np.linspace(0.5, 60.0, 400)
P = 10 ** 2.5 / (4.0 + f ** 1.7) * (1 + 3 * np.exp(-(f - 10) ** 2 / 2) + np.exp(-(f - 22) ** 2 / 8)) * np.exp(0.1 * rng.standard_normal(f.size))
P = np.array([float(f"{v:.15g}") for v in P])
out = open(sys.argv[1], "w")
out.write("# aperiodic golden: nxr.spectral.aperiodic.fit on a synthetic spectrum; see gen_aperiodic_golden.py\n")
out.write("f " + " ".join(repr(float(v)) for v in f) + "\nP " + " ".join(repr(float(v)) for v in P) + "\n")
for knee in (False, True):
    a = aperiodic.fit(P, f, knee=knee, frange=(1.0, 45.0))
    out.write(f"knee {int(knee)} {float(a['offset'])!r} {float(a['exponent'])!r} {float(a['knee'])!r} {float(a['r2'])!r} {a['n_used']}\n")
