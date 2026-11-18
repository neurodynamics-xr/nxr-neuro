"""Golden for the chunk-wise noise covariance (nxr::stats::CovarianceEstimate): synthetic channels with large DC offsets, MNE-Python's raw covariance
(compute_raw_covariance, one block, divisor N - 1) over a time window, written to cov_golden.txt.
Brainstorm's bst_noisecov needs the Brainstorm database (studies and data files), so it is not a reference here; the numbers of its divisor and DC handling are
in docs/specs/2026-11-10-neuro-forward-inverse.md section 7.
Run:  uv run --project <nxr checkout> python gen_cov_golden.py cov_golden.txt"""
import sys
import numpy as np
import mne

mne.set_log_level("ERROR")
rng = np.random.default_rng(20261111)
n_ch, n_t, sfreq = 8, 1200, 200.0
X = (np.arange(1, n_ch + 1)[:, None] * rng.standard_normal((n_ch, n_t))) + 40.0 * np.arange(n_ch)[:, None] + 1e3
X = np.array([float(f"{v:.12g}") for v in X.ravel()]).reshape(X.shape)
info = mne.create_info([f"E{i}" for i in range(n_ch)], sfreq, "eeg")
raw = mne.io.RawArray(X, info)
tmin, tmax = 0.5, 4.995          # seconds, inclusive: samples 100 .. 999
cov = mne.compute_raw_covariance(raw, tmin=tmin, tmax=tmax, tstep=None, method="empirical", reject=None, flat=None)
s0, s1 = int(round(tmin * sfreq)), int(round(tmax * sfreq)) + 1
best = None                       # which samples MNE used: the window whose direct covariance equals MNE's
for a in (s0 - 1, s0, s0 + 1):
    for b in (s1 - 1, s1, s1 + 1):
        Y = X[:, a:b]
        d = np.abs(np.cov(Y) - cov.data).max() / np.abs(cov.data).max()
        if best is None or d < best[0]:
            best = (d, a, b)
print("direct covariance of", best, "nfree", cov.nfree)
assert best[0] < 1e-9
s0, s1 = best[1], best[2]
print("window samples", s0, s1, "nfree", cov.nfree)
with open(sys.argv[1], "w") as f:
    def emit(name, a):
        a = np.atleast_2d(np.asarray(a, float))
        f.write(f"{name} {a.shape[0]} {a.shape[1]}\n" + " ".join(f"{v:.17g}" for v in a.ravel()) + "\n")
    f.write("# covariance golden: synthetic data, MNE-Python raw covariance of samples [window start, window stop); see gen_cov_golden.py\n")
    emit("X", X)
    emit("window", [s0, s1])
    emit("nsamples_mne", [cov.nfree + 1])
    emit("C_mne", cov.data)
