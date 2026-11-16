"""Writes welch_golden.txt: the input (3 channels, 2400 samples at 200 Hz, a bad span) and the Welch PSD of scipy.signal.welch, of nxr (qc.sensors.psd and
spectral.chain._welch) and of Brainstorm (bst_psd, its output only; written by gen_welch_golden.m), for test_welch.cpp. SYNTHETIC signals only.
Two stages, because Brainstorm runs in MATLAB :
  uv run --project <nxr checkout> python compute/test/golden/gen_welch_golden.py input  <work>
  matlab -batch "gen_welch_golden('<work>')"              (run from compute/test/golden)
  uv run --project <nxr checkout> python compute/test/golden/gen_welch_golden.py golden <work> compute/test/golden/welch_golden.txt"""
import sys
import numpy as np
from scipy.signal import welch

FS, T = 200.0, 2400
BAD = (900, 1250)          # samples [900, 1250) are bad


def signal():
    rng = np.random.default_rng(7)
    t = np.arange(T) / FS
    x = np.zeros((3, T))
    x[0] = 5.0 + 2.0 * np.sin(2 * np.pi * 10.0 * t) + 0.3 * rng.standard_normal(T)      # a DC offset, a 10 Hz sine of amplitude 2
    x[1] = rng.standard_normal(T) + 0.004 * np.arange(T)                                  # white noise (variance 1) on a linear trend
    x[2] = np.cumsum(rng.standard_normal(T)) * 0.05                                       # a random walk
    good = np.ones(T, bool)
    good[BAD[0]:BAD[1]] = False
    return x, good


def per_good_segments(x, good, **kw):
    """scipy on each whole segment that touches no bad sample, averaged (scipy has no mask); the segments sit where scipy's grid puts them"""
    n, hop = kw["nperseg"], kw["nperseg"] - kw["noverlap"]
    P, k = [], 0
    for s in range(0, x.shape[1] - n + 1, hop):
        if good[s:s + n].all():
            f, p = welch(x[:, s:s + n], FS, **kw)
            P.append(p)
    return f, np.mean(P, 0), len(P)


def main():
    stage, work = sys.argv[1], sys.argv[2]
    x, good = signal()
    if stage == "input":
        np.savetxt(f"{work}/input.csv", x, delimiter=",", fmt="%.17g")
        np.savetxt(f"{work}/good.csv", good.astype(int)[None, :], delimiter=",", fmt="%d")
        return
    out = open(sys.argv[3], "w")
    def w(name, A):
        A = np.atleast_2d(np.asarray(A, float))
        out.write(f"{name} {A.shape[0]} {A.shape[1]}\n" + " ".join(f"{v:.17g}" for v in A.ravel()) + "\n")
    w("x", x); w("good", good.astype(float)[None, :]); w("fs", [FS])
    # scipy: Hann (periodic), constant detrend, density; the nxr qc.sensors.psd call (4 s)
    n = 400
    f, p = welch(x, FS, window="hann", nperseg=n, noverlap=n // 2, detrend="constant", scaling="density", axis=-1)
    w("scipy_hann_const_n400", p)
    f, p, m = per_good_segments(x, good, window="hann", nperseg=n, noverlap=n // 2, detrend="constant", scaling="density")
    w("scipy_hann_const_n400_bad", p); w("scipy_hann_const_n400_bad_segments", [m])
    n = 301   # odd: no Nyquist bin
    f, p = welch(x, FS, window="hamming", nperseg=n, noverlap=100, detrend="linear", scaling="density", axis=-1)
    w("scipy_hamming_linear_n301_ov100", p)
    n = 256
    f, p = welch(x, FS, window="blackman", nperseg=n, noverlap=64, detrend=False, scaling="spectrum", axis=-1)
    w("scipy_blackman_none_spectrum_n256_ov64", p)
    from nxr.qc.sensors import psd as nxr_psd
    f, p = nxr_psd(x, FS)             # 4 s Hann, 50 %, constant, density
    w("nxr_qc_psd", p)
    # nxr.spectral.chain._welch: Hann(periodic), no detrend, n even, half overlap, across chunk edges (700-sample chunks), the diagonal of the cross-spectra
    from nxr.io.ports import ChunkSource
    from nxr.spectral.chain import _welch
    cs = ChunkSource(chunk_samples=700, read=lambda s0, s1: x[:, s0:s1].ravel())
    for tag, g in (("", None), ("_bad", good)):
        S, segs = _welch(cs, [0, 1, 2], T, FS, lambda f: np.eye(len(f)), 700, 2.0, g)     # one band per bin: the per-bin power
        w("nxr_chain" + tag, np.real(np.stack([np.diag(S[k]) for k in range(S.shape[0])], 1)))
        w("nxr_chain" + tag + "_segments", [segs])
    # Brainstorm's bst_psd (gen_welch_golden.m): the output blocks, copied (the file has the window count after the shape)
    lines = open(f"{work}/bst_out.txt").read().split("\n")
    for k in range(0, len(lines) - 1, 2):
        name, r, c, nw = lines[k].split()
        if name in ("bst_bad_2", "bst_bad_3"):
            continue        # the same as bst_bad_1: Brainstorm's bad span [start; stop] in samples is read alike for all three conventions tried
        out.write(f"{name} {r} {c}\n{lines[k + 1].strip()}\n")
        w(name + "_windows", [float(nw)])
    out.close()


main()
