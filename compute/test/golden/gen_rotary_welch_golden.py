"""Writes rotary_welch_golden.txt: a SYNTHETIC channel recording X [C x T], a pair of rows A [2 x C] (an operator), and the TWO-SIDED Welch PSD of z = A[0] X + i A[1] X
by nxr.timefreq.psd (scipy.signal.welch of a complex series, f ascending), for test_welch.cpp. Forward-rotating pair at +10 Hz (x = cos, y = sin, so z = exp(+i 2 pi f t)),
a backward pair at -7 Hz, a standing one (a line) at 5 Hz, plus noise; mixed into 4 channels by a fixed matrix, the rows the pseudo-inverse of the mixing.
  uv run --project <nxr checkout> python compute/test/golden/gen_rotary_welch_golden.py compute/test/golden/rotary_welch_golden.txt"""
import sys
import numpy as np
from nxr.timefreq import psd

FS, T = 200.0, 2400
rng = np.random.default_rng(11)
t = np.arange(T) / FS
x = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 7 * t) + 0.2 * rng.standard_normal(T)
y = np.sin(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 5 * t) - np.sin(2 * np.pi * 7 * t) + 0.2 * rng.standard_normal(T)
M = rng.standard_normal((4, 2))
X = M @ np.stack([x, y]) + 0.05 * rng.standard_normal((4, T)) + 3.0
A = np.linalg.pinv(M)                                     # [2 x 4]: the rows that recover (x, y)
z = A[0] @ X + 1j * (A[1] @ X)
out = open(sys.argv[1], "w")
def w(name, a):
    a = np.atleast_2d(np.asarray(a, float))
    out.write(f"{name} {a.shape[0]} {a.shape[1]}\n" + " ".join(f"{v:.17g}" for v in a.ravel()) + "\n")
w("X", X); w("A", A); w("fs", [FS])
f, P = psd(z, FS, nperseg=400, noverlap=200, window="hann", detrend="constant"); w("rotary_f_n400", f); w("rotary_hann_const_n400", P)
f, P = psd(z, FS, nperseg=301, noverlap=100, window="hamming", detrend="linear"); w("rotary_f_n301", f); w("rotary_hamming_linear_n301_ov100", P)
out.close()
