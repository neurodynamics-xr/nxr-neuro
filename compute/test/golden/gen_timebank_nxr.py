"""Writes timebank_nxr_golden.txt: nxr's time frames (filters/frames.py), Morlet (filters/kernels.py, spectral/joint.py), the rotary amplitudes
(spectral/rotary.py) and the joint ridge filter (spectral/joint.py) on SYNTHETIC signals, for test_timebank.cpp / test_jointbank.cpp.
Run:  uv run --project <path to nxr> python compute/test/golden/gen_timebank_nxr.py compute/test/golden/timebank_nxr_golden.txt"""
import sys
import numpy as np
from scipy.signal import hilbert
from nxr.filters import frames, kernels
from nxr.spectral import joint, rotary

out = open(sys.argv[1], "w")
def w(name, A):
    A = np.atleast_2d(np.asarray(A))
    out.write(f"{name} {A.shape[0]} {A.shape[1]}\n" + " ".join(f"{v:.15g}" for v in A.ravel()) + "\n")

fs, n = 100.0, 192
f = np.fft.rfftfreq(n, 1 / fs)
ff = np.fft.fftfreq(n, 1 / fs)
# window_frame (Meyer octave bands, Σg² = 1) and the itersine time frame (Σg² = 2)
wf = frames.window_frame(fs, n)
w("window_edges", wf.edges)
w("window_gains", frames.gains(wf, f).T)
w("window_gains_full", frames.gains(wf, np.abs(ff)).T)
it = frames.itersine_time_frame(fs, n, voices=4)
w("itersine_gains", frames.gains(it, f).T)
# Morlet hats at the Studio's centres
fc = kernels.wavelet_freqs(fs, 2, 40, 4)
w("morlet_fc", fc)
w("morlet_hat", np.stack([kernels.morlet_hat(f, c, 6) for c in fc]).T)
# nxr's Morlet coefficients of a complex series at signed frequencies
rng = np.random.default_rng(5)
t = np.arange(n) / fs
z = np.exp(2j * np.pi * 9.0 * t) * 1.5 + 0.4 * np.exp(-2j * np.pi * 9.0 * t) + 0.3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
freqs = np.array([-12.5, -5.0, 5.0, 12.5, 9.0, -9.0])
M = joint.morlet(z, fs, freqs, 6)
w("z", np.stack([z.real, z.imag]))
w("morlet_freqs", freqs)
w("morlet_re", M.real); w("morlet_im", M.imag)
# rotary amplitudes of a forward circle plus a weaker backward one (no DC, no Nyquist)
f0 = fs * 18 / n    # on a bin: no leakage into the DC and Nyquist bins, where nxr's analytic signals split a complex DC half to each sense
x = np.cos(2 * np.pi * f0 * t) * 1.2 + 0.3 * np.cos(2 * np.pi * f0 * t); y = np.sin(2 * np.pi * f0 * t) * 1.2 - 0.3 * np.sin(2 * np.pi * f0 * t)
X, Y = hilbert(x), hilbert(y)
Ap, Am = rotary.rotary_amplitudes(X[None], Y[None])
w("rot_xy", np.stack([x, y])); w("rot_Aplus", Ap); w("rot_Aminus", Am)
# ridge filter on mode pairs
P = 5
mu = np.array([0.5, 1.0, 2.0, 3.0, 4.0]); Z = rng.standard_normal((P, n)) + 1j * rng.standard_normal((P, n))
v, width = 3.0, 1.5
w("ridge_mu", mu); w("ridge_Zre", Z.real); w("ridge_Zim", Z.imag)
for s in (+1, -1):
    R = joint.ridge_filter(Z, fs, mu, v, width, s)
    w(f"ridge{s:+d}_re", R.real); w(f"ridge{s:+d}_im", R.imag)
w("ridge_params", [v, width])
out.close()
