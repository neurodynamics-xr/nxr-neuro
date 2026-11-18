"""Writes pca_golden.txt: the reference of nxr::stats, a principal component analysis by numpy's singular value decomposition of the centred table (scikit-learn's PCA with
svd_solver="full" does the same maths; scikit-learn is not needed here). Synthetic data only: three tables made from planted components plus noise.
  wide    12 x 30   (more features than observations: rank 11 after centring)
  tall    40 x 6
  scaled  15 x 8, columns of very different scales, `unit-variance` (each centred column divided by its standard deviation, ddof 1)
Per table: X, the components (rows of V^T, the first `k` of the rank), the singular values, the explained variance S^2 / (n - 1) and its ratio over the total variance, the scores U S, the mean, the scale
and the reconstructions at two k. The sign of a component is the one of its entry of largest absolute value being positive (scikit-learn's svd_flip with u_based_decision=False), applied here.
The rank is numpy's matrix_rank rule on the singular values.
Run:  python3 compute/test/golden/gen_pca_golden.py compute/test/golden/pca_golden.txt"""
import sys

import numpy as np

rng = np.random.default_rng(20261116)


def table(n, p, planted, noise, scales=None):
    U = rng.standard_normal((n, planted))
    V = rng.standard_normal((planted, p))
    X = (U * np.array([5.0, 3.0, 1.5][:planted])) @ V + noise * rng.standard_normal((n, p)) + rng.standard_normal(p) * 4
    if scales is not None:
        X = X * scales
    return X


def pca(X, unit_variance):
    n, p = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    scale = np.ones(p)
    if unit_variance:
        scale = Xc.std(axis=0, ddof=1)
        Xc = Xc / scale
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int((S > S[0] * max(n, p) * np.finfo(float).eps).sum())
    U, S, Vt = U[:, :k], S[:k], Vt[:k]
    for c in range(k):                      # the sign: the largest |loading| positive (first, if tied)
        if Vt[c, np.argmax(np.abs(Vt[c]))] < 0:
            Vt[c] *= -1
            U[:, c] *= -1
    total = (Xc ** 2).sum() / (n - 1)
    ev = S ** 2 / (n - 1)
    return dict(components=Vt, S=S, ev=ev, ratio=ev / total, scores=U * S, mean=mu, scale=scale, rank=k)


def recon(r, k):
    return (r["scores"][:, :k] @ r["components"][:k]) * r["scale"] + r["mean"]


def mat(f, name, a):
    a = np.atleast_2d(np.asarray(a, dtype=float))
    f.write(f"{name} {a.shape[0]} {a.shape[1]}\n" + " ".join(f"{v:.17g}" for v in a.ravel()) + "\n")


with open(sys.argv[1], "w") as f:
    for name, X, uv, ks in [("wide", table(12, 30, 3, 0.3), False, (2, 5)),
                            ("tall", table(40, 6, 3, 0.5), False, (1, 3)),
                            ("scaled", table(15, 8, 2, 0.4, np.array([1, 10, 100, 0.1, 5, 1, 50, 2.0])), True, (2, 4))]:
        r = pca(X, uv)
        mat(f, f"{name}_X", X)
        mat(f, f"{name}_unit_variance", [float(uv)])
        mat(f, f"{name}_rank", [r["rank"]])
        for key in ("components", "S", "ev", "ratio", "scores", "mean", "scale"):
            mat(f, f"{name}_{key}", r[key])
        mat(f, f"{name}_ks", list(ks))
        for k in ks:
            mat(f, f"{name}_recon{k}", recon(r, k))
