"""Writes taps_golden.txt: nxr's zero-phase taps and row convolution (nxr/filters/taps.py: gain_taps, convolve_rows; nxr/filters/frames.py: meyer_band),
the reference of nxr::signal, on synthetic cases. Per case: the even and the odd taps of a Meyer band, and the convolution of random rows (even and odd) over
a block's middle and against its two ends (the reflection). Numbers written exactly (repr of the double; a float32 reads back to the same bits).
Run:  uv run --project <nxr checkout> python compute/test/golden/gen_taps_golden.py compute/test/golden/taps_golden.txt"""
import sys

import numpy as np

from nxr.filters.frames import meyer_band
from nxr.filters.taps import convolve_rows, gain_taps

rng = np.random.default_rng(20261112)
out = open(sys.argv[1], "w")


def line(tag, a):
    a = np.asarray(a).ravel()
    out.write(f"{tag} {a.size} " + " ".join(repr(float(v)) for v in a) + "\n")


cases = [("alpha1200", 1200.0, 8.0, 12.0), ("beta250", 250.0, 15.0, 29.0)]
x = rng.standard_normal((3, 900)).astype(np.float32)
line("x 3 900", x)
for name, fs, lo, hi in cases:
    out.write(f"case {name} {fs!r} {lo!r} {hi!r}\n")
    for odd in (False, True):
        h = gain_taps(lambda f: meyer_band(f, lo, hi, 1 / 3), fs, odd=odd)
        kind = "odd" if odd else "even"
        line(f"taps {name} {kind}", h)
        K = len(h) - 1
        # the middle of the block (columns K .. 900 − K when they exist), and the whole block (its two ends reflected)
        left = min(K, 300)
        width = 900 - 2 * left
        line(f"conv {name} {kind} mid {left} {width}", convolve_rows(x, h, left=left, width=width, odd=odd))
        line(f"conv {name} {kind} all 0 900", convolve_rows(x, h, left=0, width=900, odd=odd))
out.close()
print(f"wrote {sys.argv[1]}")
