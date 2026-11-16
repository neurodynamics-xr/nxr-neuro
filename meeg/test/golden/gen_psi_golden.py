# Golden for nxr::meeg::phaseSlopeIndex (Nolte et al. 2008, PRL 100, 234101): synthetic Hermitian positive-semidefinite cross-spectra C(f) of M = 4 channels on F = 17 bins
# (8-12 Hz, 0.25 Hz), with per-channel delays and independent noise, and the phase-slope index of every ordered pair from MNE-Connectivity's own `_compute_psi`
# (mne-connectivity 0.9.0, the function inside `phase_slope_index`), fed the coherency C_ab / sqrt(C_aa C_bb). Run:
#   uv run --project $NXR --with mne-connectivity python gen_psi_golden.py > psi_golden.txt
# Format: "F M", the F frequencies, then C_re and C_im as F blocks of M x M, then PSI as M x M (row a, column b; diagonal 0).
import numpy as np
from mne_connectivity.effective import _compute_psi

rng = np.random.default_rng(7)
F, M = 17, 4
f = 8 + 0.25 * np.arange(F)
delay = np.array([0.0, 0.012, -0.006, 0.02])                 # s
C = np.zeros((F, M, M), complex)
for k in range(F):
    a = rng.normal(size=(M, 6)) + 1j * rng.normal(size=(M, 6))          # six "segments" of Fourier coefficients: a common source plus noise
    s = rng.normal(size=6) + 1j * rng.normal(size=6)
    X = 2.0 * np.exp(-2j * np.pi * f[k] * delay)[:, None] * s[None, :] + 0.7 * a
    C[k] = X @ X.conj().T / 6                                 # C_ab = <X_a conj(X_b)>

class Cohy:                                                   # the two things _compute_psi reads of a coherency container
    def __init__(self, d): self.d = d; self.shape = d.shape
    def get_data(self): return self.d

pairs = [(a, b) for a in range(M) for b in range(M)]
coh = np.array([[C[k, a, b] / np.sqrt(C[k, a, a].real * C[k, b, b].real) for k in range(F)] for a, b in pairs])
psi, _, _ = _compute_psi(Cohy(coh), f, [(-np.inf, np.inf)], -1)
psi = psi[:, 0].reshape(M, M)
print("# PSI golden: mne-connectivity _compute_psi on synthetic cross-spectra; see gen_psi_golden.py")
print(F, M)
print(" ".join(repr(float(x)) for x in f))
for part in (C.real, C.imag):
    print(" ".join(repr(float(x)) for x in part.ravel()))
print(" ".join(repr(float(x)) for x in psi.ravel()))
