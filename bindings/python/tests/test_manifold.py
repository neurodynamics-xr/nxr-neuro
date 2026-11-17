import numpy as np
import pytest

import nxr_neuro as nn


def icosphere(levels):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    f = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
                  (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
    v /= np.linalg.norm(v, axis=1)[:, None]
    for _ in range(levels):
        cache, verts, faces = {}, list(v), []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        v, f = np.array(verts), np.array(faces)
    return v, f.astype(np.int32)


def test_sphere_spectrum_is_l_times_l_plus_one():
    v, f = icosphere(3)                                   # 642 vertices, unit radius
    r = nn.compute.manifold.eigenmodes(v, f, 25)
    want = np.array([0] + [2] * 3 + [6] * 5 + [12] * 7 + [20] * 9)
    assert r.n_converged == 25 and r.eigenvectors.shape == (642, 25)
    np.testing.assert_allclose(r.eigenvalues[1:], want[1:], rtol=0.03)
    assert abs(r.eigenvalues[0]) < 1e-6
    assert np.all(np.diff(r.eigenvalues) >= -1e-9)


def _residual(h, u):
    """The share of h outside the span of the columns of u (0: inside; 1: orthogonal)."""
    q, _ = np.linalg.qr(u)
    return np.linalg.norm(h - q @ (q.T @ h)) / np.linalg.norm(h)


def test_sphere_modes_span_the_spherical_harmonics():
    # the vectors of a multiple eigenvalue are not unique; the subspace of each shell is
    v, f = icosphere(3)
    r = nn.compute.manifold.eigenmodes(v, f, 25, normalize=True)
    x, y, z = v.T
    lin = np.stack([x, y, z], axis=1)
    quad = np.stack([x * y, y * z, z * x, x ** 2 - y ** 2, 2 * z ** 2 - x ** 2 - y ** 2], axis=1)
    shell1, shell2 = r.eigenvectors[:, 1:4], r.eigenvectors[:, 4:9]
    assert _residual(lin, shell1) < 0.03 and _residual(quad, shell2) < 0.05
    assert _residual(lin, shell2) > 0.9


def test_remove_dc_drops_the_constant_mode():
    v, f = icosphere(2)
    r = nn.compute.manifold.eigenmodes(v, f, 10, remove_dc=True, normalize=True)
    assert r.eigenvalues[0] > 1.0
    assert r.eigenvectors.shape == (162, 9)   # the constant mode is gone


def test_mesh_errors_are_values_not_crashes():
    v, f = icosphere(1)
    with pytest.raises(ValueError, match="outside the mesh"):
        nn.compute.manifold.eigenmodes(v, np.where(f == 3, 999, f).astype(np.int32), 3)
    with pytest.raises(ValueError):
        nn.compute.manifold.eigenmodes(v[:, :2], f, 3)
    with pytest.raises(nn.Error) as e:
        nn.compute.manifold.eigenmodes(v, f, 0)
    assert e.value.code
