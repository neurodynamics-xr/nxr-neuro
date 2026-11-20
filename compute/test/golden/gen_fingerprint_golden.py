"""Writes fingerprint_golden.txt: the reference of nxr::stats' identification statistics (identification.h), made by RUNNING the authors' own notebooks on SYNTHETIC data.

The reference is the published method: da Silva Castanheira J, Orozco Perez HD, Misic B, Baillet S (2021) Brief segments of neurophysiological activity enable individual differentiation,
Nat Commun 12:5713, doi:10.1038/s41467-021-25895-8, as implemented in its authors' release megFingerprinting v1.0.0 (commit 06fdba8, GPL-3). This generator does not contain the authors'
code: it finds the release through the environment variable NXR_MEGFINGERPRINTING (a checkout of github.com/neurohazardous/megFingerprinting at 06fdba8), reads the notebooks' code
cells and executes them unchanged in a namespace it prepares (the synthetic tables in the notebooks' own variables), in a scratch folder (their figures and CSVs are written there and
thrown away). Without the variable it stops and says so; the committed golden is what the test reads.

Cells run (notebook 4_PSD_fingerprinting_within_session.ipynb):
  4   the feature vectors: each member's region x bin CSV flattened row-major           -> psd_feature check (68 x 301, two members, a closed-form spectrum)
  6   z-scores, the target x database correlation, the two accuracies
  8   the PCA reconstruction curve (scikit-learn PCA(k), its automatic solver)
  10  the reconstruction at the best k
  13  differential power (original and at the best k);  16, 18  group consistency;  20  ICC(1,1)
and cell 4 of 4_AEC_fingerprinting_within_session_fullband.ipynb: the connectome feature (68 x 68 per band, the upper triangle WITH the diagonal, six bands) -> aec_feature check.

THE PCA SOLVER. At this size (24 x 612) scikit-learn's automatic solver is RANDOMISED for k below 80 % of min(2n, p), and the release does not seed it. Here numpy's global seed is set
(SEED below) so the golden reproduces; the curve is then also run with the full solver (the same cells, PCA built with svd_solver="full"), which is the exact SVD nxr-neuro computes. The
test holds nxr-neuro to the full-solver numbers to rounding and to the release's randomised numbers within the tolerance printed here.

D_self: the notebooks' (cell 8, 10: each correlation row z-scored over all n entries, the member included, ddof 0) is the default; the paper's formula over the OTHER members (option `others`,
ddof 0) is not in the release and is computed here from the formula.

Synthetic data only: n = 12 members, p = 68 x 9 = 612 features: per member a planted profile (log-normal, so positive like a spectrum) shared by its two measurements, a group profile shared
by everyone, and independent noise per measurement.

Run (a throwaway environment with numpy, scipy, scikit-learn, pandas, matplotlib, seaborn):
    NXR_MEGFINGERPRINTING=/path/to/megFingerprinting python gen_fingerprint_golden.py fingerprint_golden.txt
"""
import contextlib
import functools
import io
import json
import os
import sys
import tempfile

import numpy as np

SEED = 20261120
N, REGIONS, BINS = 12, 68, 9


def cells(path):
    nb = json.load(open(path))
    return [("".join(c["source"]) if c["cell_type"] == "code" else None) for c in nb["cells"]]


def namespace():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy as sp
    import scipy.stats  # noqa: F401  (the notebooks call sp.stats)
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    return dict(np=np, sp=sp, pd=pd, plt=plt, sns=sns, PCA=PCA, accuracy_score=accuracy_score, listdir=os.listdir, isfile=os.path.isfile, join=os.path.join)


def run(code, ns):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, "<notebook cell>", "exec"), ns)
    import matplotlib.pyplot as plt
    plt.close("all")


def checksum(v):
    v = np.asarray(v, dtype=float).ravel()
    return [v.size, v.sum(), (np.arange(1, v.size + 1) * v).sum() / v.size]


def psd_value(m, r, b):        # a closed form the test recomputes: member m, region r, bin b
    return 1.0 + 0.01 * (m + 1) + 0.5 * np.sin(0.37 * r + 0.11 * b) + 0.001 * r * b


def aec_value(r, c, band):     # symmetric, diagonal 1
    if r == c:
        return 1.0
    i, j = min(r, c), max(r, c)
    return 0.5 * np.cos(0.13 * i + 0.07 * j + 0.5 * band)


def feature_checks(release, work):
    """cell 4 of each notebook on two members' files written in the release's layout; the vectors' checksums"""
    os.chdir(work)
    psd = cells(os.path.join(release, "4_PSD_fingerprinting_within_session.ipynb"))
    os.makedirs("output/Sources_PSD/fullband", exist_ok=True)
    names = []
    for m in range(2):
        for half in "tv":                                     # the file name's 20th character says the half (cell 4: sub[47]); characters 5:8 the member's number
            name = f"sub-0{m + 1:03d}_"
            name = name + "x" * (19 - len(name)) + half + ".csv"
            names.append(name)
            mat = np.array([[psd_value(m, r, b) for b in range(301)] for r in range(68)])
            np.savetxt("output/Sources_PSD/fullband/" + name, mat, delimiter=",", fmt="%.17g")
    ns = namespace()
    ns.update(n_subs=2, n_feats=68 * 301, n_measurements=2)
    run(psd[4], ns)
    t_psd = ns["sub_target"][:, :-1]
    assert np.array_equal(t_psd, ns["sub_database"][:, :-1])
    aec = cells(os.path.join(release, "4_AEC_fingerprinting_within_session_fullband.ipynb"))
    os.makedirs("output/csv_matrices", exist_ok=True)
    bands = ["delta", "theta", "alpha", "beta", "gamma", "hgamma"]
    for m in range(2):
        for half in "tv":
            name = f"sub-0{m + 1:03d}_"
            name = name + "x" * (19 - len(name)) + half + ".csv"
            with open("output/csv_matrices/" + name, "w") as f:
                for bi, band in enumerate(bands):
                    for r in range(68):
                        for c in range(68):
                            f.write(f"R{r:02d},R{c:02d},{aec_value(r, c, bi):.17g},{band}\n")
    ns = namespace()
    ns.update(n_subs=2, n_feats=int(68 * 69 / 2), n_measurements=2)
    run(aec[4], ns)
    t_aec = ns["sub_target"][:, :-1]
    return t_psd, t_aec


def cohort():
    rng = np.random.default_rng(SEED)
    p = REGIONS * BINS
    group = rng.standard_normal(p)
    own = rng.standard_normal((N, p))
    a = np.exp(0.6 * group + 0.5 * own + 0.25 * rng.standard_normal((N, p)))
    b = np.exp(0.6 * group + 0.5 * own + 0.25 * rng.standard_normal((N, p)))
    return a, b


def statistics(release, work, target, database, full):
    """cells 6, 8, 10, 13, 16, 18, 20 on the cohort (the label column appended, as cell 4 makes it); `full`: PCA with the full solver"""
    os.chdir(work)
    os.makedirs("output", exist_ok=True)
    psd = cells(os.path.join(release, "4_PSD_fingerprinting_within_session.ipynb"))
    ns = namespace()
    if full:
        ns["PCA"] = functools.partial(ns["PCA"], svd_solver="full")
    ns["atlasdk"] = ns["pd"].DataFrame([f"region {r}" for r in range(REGIONS)])   # cell 14 reads the atlas's names from the authors' own disk; cells 16 and 18 label their figures with it
    label = np.arange(1, N + 1, dtype=float)[:, None]
    ns.update(n_subs=N, n_feats=target.shape[1], n_measurements=2, sub_target=np.hstack([target, label]), sub_database=np.hstack([database, label]))
    np.random.seed(SEED)                                      # the release leaves scikit-learn's randomised solver unseeded; seeded here so the golden reproduces
    for i in (6, 8):
        run(psd[i], ns)
    curve = dict(steps=ns["steps"].copy(), t=ns["pca_target_from_database"].copy(), d=ns["pca_database_from_target"].copy(), self_mean=ns["pca_self_id"].mean(axis=1))
    for i in (10, 13, 16, 18, 20):
        run(psd[i], ns)
    return ns, curve


def mat(f, name, a):
    a = np.atleast_2d(np.asarray(a, dtype=float))
    f.write(f"{name} {a.shape[0]} {a.shape[1]}\n" + " ".join(f"{v:.17g}" for v in a.ravel()) + "\n")


def main():
    release = os.environ.get("NXR_MEGFINGERPRINTING")
    if not release or not os.path.isdir(release):
        sys.exit("gen_fingerprint_golden: set NXR_MEGFINGERPRINTING to a checkout of the authors' release (megFingerprinting v1.0.0, 06fdba8); the committed golden is unchanged")
    out = os.path.abspath(sys.argv[1])
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as work:
        t_psd, t_aec = feature_checks(release, work)
        target, database = cohort()
        nb, c_nb = statistics(release, work, target, database, full=False)
        ex, c_ex = statistics(release, work, target, database, full=True)
    os.chdir(here)
    pred = nb["predictions"]
    n = pred.shape[0]
    others = np.array([(pred[i, i] - np.delete(pred[i], i).mean()) / np.delete(pred[i], i).std() for i in range(n)])
    best_nb, best_ex = int(np.argmax(c_nb["self_mean"])), int(np.argmax(c_ex["self_mean"]))
    tol = dict(curve_self=float(np.abs(c_nb["self_mean"] - c_ex["self_mean"]).max()),
               icc=float(np.abs(nb["pca_icc"] - ex["pca_icc"]).max()), gc=float(np.abs(nb["pca_group_consistency"] - ex["pca_group_consistency"]).max()),
               dp=float(np.abs(nb["pca_differential_power"] - ex["pca_differential_power"]).max()))
    print("randomised vs full solver: max |difference|", tol, "; best k", c_nb["steps"][best_nb], "vs", c_ex["steps"][best_ex])
    with open(out, "w") as f:
        mat(f, "psd_feature", checksum(t_psd[1]))            # member 2 (m = 1)
        mat(f, "aec_feature", checksum(t_aec[1]))
        mat(f, "target", target)
        mat(f, "database", database)
        mat(f, "correlation", pred)
        mat(f, "accuracy", [nb["target_from_database"], nb["database_from_target"]])
        mat(f, "self_all", np.diagonal(__import__("scipy").stats.zscore(pred, axis=1)))
        mat(f, "self_others", others)
        mat(f, "icc", nb["icc"])
        mat(f, "differential_power", nb["differential_power"])
        mat(f, "group_consistency", nb["group_consistency"])
        mat(f, "steps", c_nb["steps"])
        for tag, c, ns, best in (("full", c_ex, ex, best_ex), ("release", c_nb, nb, best_nb)):
            mat(f, f"curve_{tag}", np.vstack([c["t"], c["d"], c["self_mean"]]))
            mat(f, f"best_{tag}", [best])
            mat(f, f"best_accuracy_{tag}", [ns["pca_target_from_database"], ns["pca_database_from_target"]])
            mat(f, f"best_self_{tag}", ns["pca_self_id"])
            mat(f, f"best_icc_{tag}", ns["pca_icc"])
            mat(f, f"best_differential_power_{tag}", ns["pca_differential_power"])
            mat(f, f"best_group_consistency_{tag}", ns["pca_group_consistency"])


if __name__ == "__main__":
    main()
