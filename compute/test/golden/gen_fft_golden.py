"""Writes fft_golden.txt: random inputs and numpy.fft's outputs, for test_fft.cpp. Synthetic data only.
Run:  uv run --project <nxr checkout> python compute/test/golden/gen_fft_golden.py compute/test/golden/fft_golden.txt"""
import sys
import numpy as np

rng = np.random.default_rng(20261110)
out = open(sys.argv[1], "w")

def flat(a):
    a = np.asarray(a)
    v = np.stack([a.real, a.imag], -1).ravel() if np.iscomplexobj(a) else a.ravel()
    return " ".join(f"{x:.17g}" for x in v)

def emit(kind, norm, axis, shape, x, y):
    out.write(f"case {kind} {norm} {axis} {len(shape)} {' '.join(map(str, shape))}\n")
    out.write(f"in {flat(x)}\nout {flat(y)}\n")

for norm in ("backward", "ortho", "forward"):
    for n in (1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 30, 64, 97, 100, 257):
        if norm != "backward" and n not in (8, 17, 100):
            continue
        z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        emit("c2c_fwd", norm, 0, (n,), z, np.fft.fft(z, norm=norm))
        emit("c2c_inv", norm, 0, (n,), z, np.fft.ifft(z, norm=norm))
        x = rng.standard_normal(n)
        emit("r2c", norm, 0, (n,), x, np.fft.rfft(x, norm=norm))
        X = np.fft.rfft(x)                       # a Hermitian half-spectrum
        emit("c2r", norm, 0, (n,), X, np.fft.irfft(X, n, norm=norm))
    for shape, axis in (((3, 8), 0), ((3, 8), 1), ((4, 7, 6), 1), ((4, 7, 6), 2), ((5, 9), 0)):
        if norm != "backward" and shape != (4, 7, 6):
            continue
        z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        emit("c2c_fwd", norm, axis, shape, z, np.fft.fft(z, axis=axis, norm=norm))
        x = rng.standard_normal(shape)
        emit("r2c", norm, axis, shape, x, np.fft.rfft(x, axis=axis, norm=norm))
        X = np.fft.rfft(x, axis=axis)
        emit("c2r", norm, axis, shape, X, np.fft.irfft(X, shape[axis], axis=axis, norm=norm))
# helpers
for n, d in ((8, 1.0), (9, 0.25), (10, 2.0), (1, 1.0)):
    out.write(f"freq {n} {d:.17g}\nfftfreq {flat(np.fft.fftfreq(n, d))}\nrfftfreq {flat(np.fft.rfftfreq(n, d))}\n")
out.close()
