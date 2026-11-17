import numpy as np
import pytest

import nxr_neuro as nn

pca = nn.ops.pca


def reference(X, unit=False):
    mu = X.mean(axis=0)
    Xc = X - mu
    sc = Xc.std(axis=0, ddof=1) if unit else np.ones(X.shape[1])
    U, S, Vt = np.linalg.svd(Xc / sc, full_matrices=False)
    idx = np.argmax(np.abs(Vt), axis=1)
    sign = np.sign(Vt[np.arange(Vt.shape[0]), idx])
    return Vt * sign[:, None], S, (U * S) * sign, mu, sc


@pytest.mark.parametrize("unit", [False, True])
def test_pca_equals_the_svd_with_the_sign_convention(rng, unit):
    X = rng.normal(size=(40, 12)) @ rng.normal(size=(12, 12)) + 3.0
    r = pca.pca(X, scale="unit_variance" if unit else "none")
    comp, S, scores, mu, sc = reference(X, unit)
    k = r.rank
    assert k == 12
    np.testing.assert_allclose(r.components, comp, atol=1e-10)
    np.testing.assert_allclose(r.singular_values, S, rtol=1e-10)
    np.testing.assert_allclose(r.scores, scores, atol=1e-9)
    np.testing.assert_allclose(r.mean, mu)
    np.testing.assert_allclose(r.scale, sc)
    np.testing.assert_allclose(r.explained_variance, S ** 2 / 39, rtol=1e-10)
    assert r.explained_variance_ratio.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(r.components @ r.components.T, np.eye(k), atol=1e-10)


def test_rank_is_at_most_n_minus_one(rng):
    X = rng.normal(size=(6, 20))
    assert pca.pca(X).rank == 5


def test_reconstruct(rng):
    X = rng.normal(size=(30, 8)) + 5
    r = pca.pca(X)
    np.testing.assert_allclose(pca.reconstruct(r.scores, r.components, r.mean, r.scale, r.rank), X, atol=1e-10)
    np.testing.assert_allclose(pca.reconstruct(r.scores, r.components, r.mean, r.scale, 0), np.tile(r.mean, (30, 1)), atol=1e-12)
    # the best rank-2 approximation: the error is the discarded singular values
    err = np.linalg.norm(X - pca.reconstruct(r.scores, r.components, r.mean, r.scale, 2))
    assert err == pytest.approx(np.sqrt((r.singular_values[2:] ** 2).sum()), rel=1e-9)
    with pytest.raises(ValueError):
        pca.reconstruct(r.scores, r.components, r.mean, r.scale, 99)


def test_pca_precision_and_refusals(rng):
    X = rng.normal(size=(20, 5)).astype(np.float32)
    r = pca.pca(X)
    assert r.components.dtype == np.float32 and r.scores.dtype == np.float32
    bad = X.astype(np.float64)
    bad[3, 2] = np.nan
    with pytest.raises(ValueError):
        pca.pca(bad)
    const = np.ones((10, 3))
    const[:, 0] = np.arange(10)
    with pytest.raises(ValueError):
        pca.pca(const, scale="unit_variance")        # a constant column
