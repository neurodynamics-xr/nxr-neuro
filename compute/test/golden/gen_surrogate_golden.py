"""Writes surrogate_golden.txt: the reference of nxr::surrogate, nxr's `lagfree_real` and `envelope_matched_null` (nxr/detect/vortex.py), on a SYNTHETIC 3-channel recording at 100 Hz (4096 samples): a 10 Hz source (a
resonance, 3 Hz wide) reaching the channels with LAGS of 0, 2 and 4 samples (so the data has a large imaginary cross-spectrum) and a slow amplitude modulation (0.4 Hz). Run:
  uv run --project <nxr checkout> python compute/test/golden/gen_surrogate_golden.py compute/test/golden/surrogate_golden.txt
Written: the recording X [3 x 4096]; then nxr's numbers, each the mean over 40 draws (seeds 0..39) of the SAME statistic the C++ test computes on its own draws (a Welch estimate, 256-sample Hann segments, half overlap):
  spec   the Bhattacharyya agreement between the draw's mean power spectrum (channel mean) and the data's, 2-30 Hz
  imre   Im / sqrt(P0 P1) and Re / sqrt(P0 P1) of the draw's cross-spectrum of channels 0 and 1 in 6-14 Hz (the data's, for reference, has a large Im)
  cov    the draw's covariance matrix entries (00 01 02 11 12 22) and the data's
  env    the correlation of the draw's one-cycle RMS envelope with the data's, and the ratio of the two series' means (envelope_matched_null)"""
import sys

import numpy as np

from nxr.detect.vortex import envelope_matched_null, lagfree_real

FS, N, SEG = 100.0, 4096, 256
rng0 = np.random.default_rng(20261115)
# a resonant source: white noise through a 10 Hz resonance, then the channels' lags and a little sensor noise
white = rng0.standard_normal(N + 64)
f = np.fft.rfftfreq(N + 64, 1 / FS)
src = np.fft.irfft(np.fft.rfft(white) / (1 + ((f - 10) / 1.5) ** 2) ** 0.5, n=N + 64)
t = np.arange(N) / FS
env = 1 + 0.5 * np.cos(2 * np.pi * 0.4 * t)
X = np.stack([src[32 - d: 32 - d + N] * env * g + 0.05 * rng0.standard_normal(N) for d, g in ((0, 1.0), (2, 0.8), (4, 0.6))])
X -= X.mean(1, keepdims=True)


def welch(x):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(SEG) / SEG)
    segs = [np.fft.rfft(x[:, s:s + SEG] * w, axis=1) for s in range(0, x.shape[1] - SEG + 1, SEG // 2)]
    P = np.mean([np.abs(a) ** 2 for a in segs], axis=0)
    S01 = np.mean([np.conj(a[0]) * a[1] for a in segs], axis=0)
    return P, S01


def stats(x):
    P, S01 = welch(x)
    fr = np.fft.rfftfreq(SEG, 1 / FS)
    return P, S01, fr


Pd, _, fr = stats(X)
m = (fr >= 2) & (fr <= 30); b = (fr >= 6) & (fr <= 14)


def bhatt(P):
    a, c = Pd.mean(0)[m], P.mean(0)[m]
    return float(np.sum(np.sqrt(a * c)) / np.sqrt(a.sum() * c.sum()))


def imre(P, S01):
    p = np.sqrt(P[0][b] * P[1][b]).sum()
    return float(S01[b].imag.sum() / p), float(S01[b].real.sum() / p)


def one_cycle(v, per=10):
    return np.sqrt(np.convolve((v ** 2).mean(0), np.ones(per) / per, "same"))


SPEC, IM, RE, COV, CORR, RAT = [], [], [], [], [], []
for seed in range(40):
    Y = lagfree_real(X, None, np.random.default_rng(seed)).astype(float)
    P, S01, _ = stats(Y)
    SPEC.append(bhatt(P)); i, r = imre(P, S01); IM.append(i); RE.append(r)
    C = np.cov(Y); COV.append([C[0, 0], C[0, 1], C[0, 2], C[1, 1], C[1, 2], C[2, 2]])
    E = envelope_matched_null(X, np.ones(N, bool), 10, np.random.default_rng(seed)).astype(float)
    ed, en = one_cycle(X), one_cycle(E)
    CORR.append(float(np.corrcoef(ed, en)[0, 1])); RAT.append(float(en.mean() / ed.mean()))
Cd = np.cov(X); Pd_, S01d, _ = stats(X); imd, red = imre(Pd_, S01d)
with open(sys.argv[1], "w") as o:
    o.write(f"{X.shape[0]} {X.shape[1]}\n")
    for row in X: o.write(" ".join(f"{v:.7g}" for v in row) + "\n")
    o.write("spec %.6f\n" % np.mean(SPEC))
    o.write("imre %.6f %.6f\n" % (np.mean(IM), np.mean(RE)))
    o.write("imre_data %.6f %.6f\n" % (imd, red))
    o.write("cov " + " ".join("%.6f" % v for v in np.mean(COV, 0)) + "\n")
    o.write("cov_data " + " ".join("%.6f" % v for v in (Cd[0, 0], Cd[0, 1], Cd[0, 2], Cd[1, 1], Cd[1, 2], Cd[2, 2])) + "\n")
    o.write("env %.6f %.6f\n" % (np.mean(CORR), np.mean(RAT)))
print("spec", np.mean(SPEC), "imre", np.mean(IM), np.mean(RE), "data", imd, red, "env", np.mean(CORR), np.mean(RAT))
