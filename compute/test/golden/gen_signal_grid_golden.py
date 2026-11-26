"""Writes signal_grid_golden.txt for test_signal_grid.cpp. Synthetic data only. No MATLAB: each reference is NumPy/SciPy applied to the published formula.
  wgft / iwgft   Shuman, Ricaud & Vandergheynst 2016: atoms g_{i,k} = M_k T_i g from numpy.linalg.eigh of a small graph's Laplacian (the eigenvectors are written, so the signs agree)
  jft / ijft     GSPBox gsp_jft: graph Fourier transform, then numpy.fft.fft(axis=1, norm='ortho')
  a bank on the Chebyshev bound   the true lambda_max (numpy.linalg.eigvalsh) the Lanczos estimate must bound
  xwt / wcoherence  Grinsted, Moore & Jevrejeva 2004 (wtc.m, smoothwavelet.m) written out in NumPy
Run:  python3 compute/test/golden/gen_signal_grid_golden.py compute/test/golden/signal_grid_golden.txt"""
import sys
import numpy as np

rng = np.random.default_rng(20261118)
out = open(sys.argv[1], "w")

def put(key, a):
    a = np.asarray(a)
    v = np.stack([a.real, a.imag], -1).ravel() if np.iscomplexobj(a) else a.ravel()
    out.write(key + " " + " ".join(f"{x:.17g}" for x in v) + "\n")

# a small weighted graph: a ring of 12 vertices with chords
N = 12
W = np.zeros((N, N))
for i in range(N):
    j = (i + 1) % N
    W[i, j] = W[j, i] = 1 + 0.5 * rng.random()
for i, j in ((0, 5), (2, 8), (3, 10)):
    W[i, j] = W[j, i] = 0.7
L = np.diag(W.sum(1)) - W
lam, phi = np.linalg.eigh(L)
put("graph.L", L); put("graph.lambda", lam); put("graph.phi", phi)       # phi column-major order is not assumed: row-major N x N written, read as such
put("graph.lmax", np.array([lam.max()]))

# windowed graph Fourier transform, kernel g^(lambda) = exp(-0.8 lambda)
def kern(l): return np.exp(-0.8 * l)
f = rng.standard_normal(N)
gh = kern(lam)
T = np.sqrt(N) * (phi * gh) @ phi.T                  # T[n, i] = (T_i g)(n)
S = np.zeros((N, N))
for i in range(N):
    for k in range(N):
        atom = np.sqrt(N) * phi[:, k] * T[:, i]       # g_{i,k}(n) = (M_k T_i g)(n)
        S[i, k] = f @ atom
put("wgft.f", f); put("wgft.S", S)
den = N * (T ** 2).sum(1)                              # N sum_i (T_i g)(m)^2
num = np.zeros(N)
for i in range(N):
    for k in range(N):
        num += S[i, k] * np.sqrt(N) * phi[:, k] * T[:, i]
put("wgft.back", num / den)

# jft: [vertices x T]; real and complex
Tt = 16
F = rng.standard_normal((N, Tt))
put("jft.F", F); put("jft.Fhat", np.fft.fft(phi.T @ F, axis=1, norm="ortho"))
Fc = rng.standard_normal((N, Tt)) + 1j * rng.standard_normal((N, Tt))
put("jft.Fc", Fc); put("jft.Fchat", np.fft.fft(phi.T @ Fc, axis=1, norm="ortho"))

# xwt and wcoherence (Grinsted): smoothing in time (Gaussian at each scale, FFT padded to 2^ceil(log2 n)) then across scales (boxcar kernel, conv2 'same')
def smooth(Wm, dt, scale, steps):
    S_, n = Wm.shape
    npad = 2 ** int(np.ceil(np.log2(n)))
    k = 2 * np.pi * np.fft.fftfreq(npad, dt)
    res = np.zeros_like(Wm)
    for s in range(S_):
        Fg = np.exp(-0.5 * (scale[s] * k) ** 2)
        res[s] = np.fft.ifft(Fg * np.fft.fft(Wm[s], npad))[:n]
    if steps > 0:
        frac = steps % 1   # the kernel's width uses MATLAB's round (halves away from zero)
        ker = np.concatenate(([frac], np.ones(max(0, 2 * int(np.floor(steps + 0.5)) - 1)), [frac]))
        ker = ker / ker.sum()
        L_ = len(ker); off = L_ // 2
        sm = np.zeros_like(res)
        for t in range(n):
            full = np.convolve(res[:, t], ker, mode="full")
            sm[:, t] = full[off:off + S_]
        res = sm
    return res

for name, (S_, n, dt, dj, steps) in {"a": (8, 50, 0.1, 0.25, 1.2), "b": (6, 64, 0.02, 0.5, 0.6 / (2 * 0.5)), "c": (5, 33, 0.1, 0.25, 2.5), "d": (4, 40, 0.05, 1.0, 0.0)}.items():
    Wx = rng.standard_normal((S_, n)) + 1j * rng.standard_normal((S_, n))
    Wy = 0.6 * Wx + 0.8 * (rng.standard_normal((S_, n)) + 1j * rng.standard_normal((S_, n)))
    scale = 0.2 * 2.0 ** (dj * np.arange(S_))
    sinv = (1.0 / scale)[:, None]
    cross = Wx * np.conj(Wy)
    R2 = np.abs(smooth(sinv * cross, dt, scale, steps)) ** 2 / (np.real(smooth(sinv * np.abs(Wx) ** 2, dt, scale, steps)) * np.real(smooth(sinv * np.abs(Wy) ** 2, dt, scale, steps)))
    put(f"wcoh.{name}.dims", np.array([S_, n, dt, steps]))
    put(f"wcoh.{name}.scale", scale)
    put(f"wcoh.{name}.Wx", Wx); put(f"wcoh.{name}.Wy", Wy)      # row-major [scales x time]
    put(f"wcoh.{name}.xwt", cross); put(f"wcoh.{name}.R2", R2)
