"""Writes morlet_golden.txt: nxr's Morlet band (the project's fixed-band rule), the reference of nxr::signal::timefilterbank::morletBand and of the pipelines' `shape: morlet`.
Per case (lo, hi Hz at the rate fs): the gain G(f) = morlet_gain(f, centre, fwhm)/2 on a grid (nxr/detect/vortex.py `morlet_gain`, the Studio's kernels.morlet_hat; the
centre is the ARITHMETIC mean (lo + hi)/2 and the FWHM is hi − lo: `bands_fixed`), the even and odd taps of that gain (nxr/filters/taps.py gain_taps), and the band's
analytic signal of a synthetic series by nxr's FFT route (`morlet_gain` on the FFT of a series, ifft), the reference of the taps route.
Synthetic data only. Run:  uv run --project <nxr checkout> python compute/test/golden/gen_morlet_golden.py compute/test/golden/morlet_golden.txt"""
import sys

import numpy as np

from nxr.detect.vortex import morlet_gain
from nxr.filters.kernels import morlet_hat
from nxr.filters.taps import gain_taps

out = open(sys.argv[1], "w")


def line(tag, a):
    a = np.asarray(a).ravel()
    if np.iscomplexobj(a):
        a = np.stack([a.real, a.imag], -1).ravel()
    out.write(f"{tag} {a.size} " + " ".join(repr(float(v)) for v in a) + "\n")


rng = np.random.default_rng(20261114)
n = 4096
x = rng.standard_normal(n)
line("x 1", x)
for name, fs, lo, hi in [("alpha100", 100.0, 8.0, 12.0), ("beta100", 100.0, 15.0, 29.0)]:
    f0, fwhm = (lo + hi) / 2, hi - lo
    cycles = f0 * 2 * np.sqrt(2 * np.log(2)) / fwhm
    out.write(f"case {name} {fs!r} {lo!r} {hi!r}\n")
    fg = np.arange(0, 60.5, 0.5)
    line(f"gain {name}", morlet_gain(fg[1:], f0, fwhm) / 2)         # f > 0 (the f = 0 bin is gated off in nxr; the continuous gain is used there)
    G = lambda f: morlet_hat(f, f0, cycles)
    line(f"taps {name} even", gain_taps(G, fs, odd=False))
    line(f"taps {name} odd", gain_taps(G, fs, odd=True))
    xa = np.fft.ifft(np.fft.fft(x) * morlet_gain(np.fft.fftfreq(n, 1 / fs), f0, fwhm))
    line(f"analytic {name}", xa)
out.close()
print(f"wrote {sys.argv[1]}")
