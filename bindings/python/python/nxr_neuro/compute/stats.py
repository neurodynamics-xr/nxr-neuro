"""Principal components of a table of observations x features (`nxr/stats/pca.h`), by the thin singular value decomposition (scikit-learn's PCA, svd_solver="full").

Each column's mean is removed; optionally each column is then divided by its standard deviation (ddof 1; the correlation matrix's components). Never
the p x p covariance matrix: p can be tens of thousands where n is a few hundred. Every component up to the rank of the centred table is returned.

Sign convention (fixed; scikit-learn's svd_flip with u_based_decision=False): each component's sign is chosen so that its entry of largest absolute
value (the first, if several are equal) is positive; the scores' column takes the same sign.

What it cannot show: a component is a direction of variance in THIS sample; with n observations at most n - 1 components exist, and the leading ones
of a small sample are not stable estimates of a population's.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .. import _core


class PcaResult(NamedTuple):
    components: np.ndarray                 # [k, p], unit rows, the sign convention above
    singular_values: np.ndarray            # [k], of the centred (and scaled) table, descending
    explained_variance: np.ndarray         # [k], singular value^2 / (n - 1)
    explained_variance_ratio: np.ndarray   # [k], over the table's total variance (sums to 1 at full rank)
    scores: np.ndarray                     # [n, k], U S
    mean: np.ndarray                       # [p], the columns' means removed
    scale: np.ndarray                      # [p], the divisor of each column after centring (all 1 for scale="none")
    rank: int                              # k


def pca(x, scale: str = "none") -> PcaResult:
    """The components of x [n, p], n >= 2 (`nxr::stats::pca`).

    Parameters
    ----------
    x : array [n, p]
        Observations by features, float32 or float64. A non-finite value, a table with no variance, or (scale="unit_variance") a constant column is
        refused with an error saying why.
    scale : {"none", "unit_variance"}
        "none": the columns keep their units (the covariance's components; default). "unit_variance": divided by their standard deviation after
        centring (ddof 1; the correlation's components).

    Returns
    -------
    PcaResult
        components, singular_values, explained_variance, explained_variance_ratio, scores, mean, scale, rank. Matches scikit-learn's PCA(svd_solver="full")
        (sign aside, which the convention above fixes).
    """
    return PcaResult(*_core.stats_pca(x, scale))


def reconstruct(scores, components, mean, scale, k: int):
    """The rank-k reconstruction scores[:, :k] @ components[:k] * scale + mean, [n, p] (`nxr::stats::reconstruct`).

    The best rank-k approximation of the centred table in the least-squares sense (Eckart-Young). k from 0 (the mean) to the components held.
    """
    return _core.stats_reconstruct(scores, components, mean, scale, int(k))
