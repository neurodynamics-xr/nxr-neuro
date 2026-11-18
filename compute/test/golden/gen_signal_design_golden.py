"""Writes signal_design_golden.txt: SciPy's outputs for fir1 (scipy.signal.firwin), butter (scipy.signal.butter), mscohere (scipy.signal.coherence),
xcorr (scipy.signal.correlate + correlation_lags) and istft (scipy.signal.istft), for test_signal_design.cpp. Synthetic data only.
Run:  python3 compute/test/golden/gen_signal_design_golden.py compute/test/golden/signal_design_golden.txt
Format: one line per array, `key v v v ...` (complex numbers as re im pairs)."""
import sys
import numpy as np
from scipy import signal

rng = np.random.default_rng(20261118)
out = open(sys.argv[1], "w")

def put(key, a):
    a = np.asarray(a)
    v = np.stack([a.real, a.imag], -1).ravel() if np.iscomplexobj(a) else a.ravel()
    out.write(key + " " + " ".join(f"{x:.17g}" for x in v) + "\n")

# fir1: MATLAB fir1(n, Wn, type, window) = scipy firwin(n+1, Wn, window, pass_zero) with the passband-centre scaling
for name, order, wn, pz in (("low", 30, [0.3], True), ("high", 30, [0.4], False), ("bandpass", 40, [0.2, 0.5], False), ("stop", 40, [0.2, 0.5], True), ("low_odd_order", 21, [0.15], True)):
    for wname in ("hamming", "hann", "blackman"):
        put(f"fir1.{name}.{wname}", signal.firwin(order + 1, wn, window=wname, pass_zero=pz, scale=True))
put("fir1.bandpass.noscale.kaiser5", signal.firwin(41, [0.2, 0.5], window=("kaiser", 5.0), pass_zero=False, scale=False))

# butter (output 'ba')
for name, n, wn, bt in (("low", 4, [0.25], "lowpass"), ("high", 5, [0.4], "highpass"), ("bandpass", 3, [0.2, 0.45], "bandpass"), ("stop", 4, [0.15, 0.5], "bandstop"), ("low1", 1, [0.1], "lowpass"), ("low8", 8, [0.3], "lowpass")):
    b, a = signal.butter(n, wn, btype=bt)
    put(f"butter.{name}.b", b); put(f"butter.{name}.a", a)

# mscohere (scipy.signal.coherence): two noisy signals sharing a 12 Hz tone; the frequency axis; several windows
fs = 100.0
t = np.arange(2000) / fs
common = np.sin(2 * np.pi * 12 * t)
x = common + 0.8 * rng.standard_normal(t.size)
y = 0.7 * np.roll(common, 3) + 0.8 * rng.standard_normal(t.size) + 0.05 * t
put("coh.x", x); put("coh.y", y)
for name, nper, nov, nfft, det in (("a", 256, 128, None, "constant"), ("b", 200, 150, 256, "linear"), ("c", 128, 0, None, False), ("d", 255, 100, None, "constant")):
    win = signal.get_window("hann", nper)          # scipy's default: periodic Hann
    f, c = signal.coherence(x, y, fs=fs, window=win, nperseg=nper, noverlap=nov, nfft=nfft, detrend=det)
    put(f"coh.{name}.f", f); put(f"coh.{name}.C", c)
    put(f"coh.{name}.window", win)

# xcorr: scipy.signal.correlate(x, y, 'full') with correlation_lags; real, complex, unequal lengths
xr = rng.standard_normal(37); yr = rng.standard_normal(37); ys = rng.standard_normal(23)
xc = rng.standard_normal(29) + 1j * rng.standard_normal(29); yc = rng.standard_normal(29) + 1j * rng.standard_normal(29)
for key, a in (("xr", xr), ("yr", yr), ("ys", ys), ("xc", xc), ("yc", yc)):
    put("xc." + key, a)
put("xc.real", signal.correlate(xr, yr, mode="full", method="direct"))
put("xc.short", signal.correlate(xr, ys, mode="full", method="direct")); put("xc.short.lags", signal.correlation_lags(37, 23))
put("xc.complex", signal.correlate(xc, yc, mode="full", method="direct"))
put("xc.auto", signal.correlate(xr, xr, mode="full", method="direct"))

# istft: Zxx from scipy.signal.stft, inverse by scipy.signal.istft (the spec the C++ must follow)
sig = rng.standard_normal(1000) + np.sin(2 * np.pi * 0.05 * np.arange(1000))
for name, nper, nov, nfft, bnd, scaling in (("a", 128, 64, None, True, "spectrum"), ("b", 100, 75, 128, True, "spectrum"), ("c", 64, 48, None, False, "spectrum"), ("d", 128, 64, None, True, "psd"), ("e", 127, 63, None, True, "spectrum")):
    win = signal.get_window("hann", nper)
    f, tt, Z = signal.stft(sig, fs=2.0, window=win, nperseg=nper, noverlap=nov, nfft=nfft, boundary="zeros" if bnd else None, padded=bnd, scaling=scaling)
    _, xr_ = signal.istft(Z, fs=2.0, window=win, nperseg=nper, noverlap=nov, nfft=nfft, boundary=bnd, scaling=scaling)
    put(f"istft.{name}.shape", np.array([Z.shape[0], Z.shape[1], nper, nov, nfft or nper, int(bnd)]))
    put(f"istft.{name}.window", win)
    put(f"istft.{name}.Z", Z.T)        # time-major rows so the C++ can read column by column
    put(f"istft.{name}.x", xr_)
# a two-sided (complex-input) case
zc = rng.standard_normal(400) + 1j * rng.standard_normal(400)
win = signal.get_window("hann", 64)
f, tt, Z = signal.stft(zc, fs=1.0, window=win, nperseg=64, noverlap=32, return_onesided=False, boundary="zeros", padded=True)
_, x2 = signal.istft(Z, fs=1.0, window=win, nperseg=64, noverlap=32, input_onesided=False, boundary=True)
put("istft.two.shape", np.array([Z.shape[0], Z.shape[1], 64, 32, 64, 1])); put("istft.two.window", win); put("istft.two.Z", Z.T); put("istft.two.x", x2)
