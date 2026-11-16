"""Writes burst_stages_golden.txt: the reference of nxr::risefall::stagesOf and passesGate (global_bursts' `fit_gate` and `stages`), nxr's `vortex.burst_stages` rule on SYNTHETIC amplitude series:
the Model "quadrature" fit of `vortex.risefall` (margin 10 cycles, smoothing 1 cycle, tau <= 10 cycles; `burst_stages`' own calls) of each episode, its build-up / plateau / decay windows
(onset tOn ± 2 tauR, within, offset tOff ± 2 tauF) and the gate (ok, not at its bound, r2 >= 0.7). Episodes: a clean one, one with a long fall (tau at its bound), one in noise (low r2).
Run:  uv run --project <nxr checkout> python compute/test/golden/gen_burst_stages_golden.py compute/test/golden/burst_stages_golden.txt"""
import sys

import numpy as np
from nxr.detect.vortex import risefall

fs, f0 = 100.0, 10.0
cyc = 1 / f0
rng = np.random.default_rng(11)
t = np.arange(int(40 * fs)) / fs
S = lambda x: 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def episode(t_on, t_off, tau_r, tau_f, base, height, noise):
    a = np.sqrt(base ** 2 + (height * S((t - t_on) / tau_r) * S(-(t - t_off) / tau_f)) ** 2)
    return a + noise * rng.standard_normal(t.size)


cases = [("clean", episode(10.0, 14.0, 0.2, 0.3, 0.2, 1.0, 0.01), (9.0, 15.0)),
         ("long_fall", episode(10.0, 14.0, 0.2, 5.0, 0.2, 1.0, 0.01), (9.0, 16.0)),
         ("noise", episode(10.0, 12.0, 0.3, 0.3, 0.3, 0.15, 0.15), (9.5, 12.5))]
with open(sys.argv[1], "w") as out:
    out.write(f"rate {fs!r} cycle {cyc!r} margin 10 smooth 1 tau_max 10 min_r2 0.7 n {t.size}\n")
    for name, a, (t0, t1) in cases:
        R = risefall(a, fs, t0, t1, margin=10 * cyc, smooth=cyc, tau_max=10 * cyc)
        gate = bool(R["ok"] and not R["at_bound"] and R["r2"] >= 0.7)
        out.write(f"case {name} {t0!r} {t1!r}\n")
        out.write("series " + " ".join(repr(float(v)) for v in a) + "\n")
        f = lambda v: "nan" if v is None else repr(float(v))
        pl = R["plateau"]
        out.write(f"fit ok {int(R['ok'])} at_bound {int(R['at_bound'])} gate {int(gate)} r2 {f(R['r2'])} tOn {f(R['tOn'])} tOff {f(R['tOff'])} tauR {f(R['tauR'])} tauF {f(R['tauF'])}\n")
        out.write("onset " + " ".join(f(v) for v in R["buildup"]) + "\n" if R["buildup"] else "onset nan nan\n")
        out.write("within " + (" ".join(f(v) for v in pl) if pl else "nan nan") + "\n")
        out.write("offset " + " ".join(f(v) for v in R["decay"]) + "\n" if R["decay"] else "offset nan nan\n")
        print(name, "ok", R["ok"], "at_bound", R["at_bound"], "r2", R["r2"], "gate", gate)
