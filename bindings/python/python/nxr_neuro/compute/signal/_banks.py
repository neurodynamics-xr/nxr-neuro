"""Argument conversion for the bank functions: a bank is a dict of the library's design fields; this turns NumPy values into plain ones and builds the sparse
triplets the native layer takes. No maths."""

from __future__ import annotations

import numpy as np


def plain(bank) -> dict:
    """A bank dict with NumPy arrays as lists, tuples as lists, and unset (None) fields dropped; anything else as given."""
    if not hasattr(bank, "items"):
        raise TypeError("a bank is a dict of the design fields (family, sides, ...)")
    out = {}
    for k, v in bank.items():
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, (tuple, list)):
            v = [list(x) if isinstance(x, (tuple, np.ndarray)) else x for x in v]
            if k in ("limits", "scale_limits"):
                v = tuple(v)
        out[k] = v
    return out


def sparse(m):
    """(row, col, value, n_rows, n_cols) of a scipy sparse matrix, a dense matrix, or a 1-D array taken as a diagonal; None stays None."""
    if m is None:
        return None
    if hasattr(m, "tocoo"):
        c = m.tocoo()
        return (c.row.astype(np.int64), c.col.astype(np.int64), c.data.astype(np.float64), int(c.shape[0]), int(c.shape[1]))
    a = np.asarray(m, dtype=np.float64)
    if a.ndim == 1:
        i = np.arange(a.size, dtype=np.int64)
        return (i, i.copy(), a.copy(), a.size, a.size)
    if a.ndim == 2:
        r, c = np.nonzero(a)
        return (r.astype(np.int64), c.astype(np.int64), a[r, c].copy(), a.shape[0], a.shape[1])
    raise ValueError("a mass or laplacian is a sparse matrix, a matrix, or a diagonal as a 1-D array")


def vector(v):
    return None if v is None else np.ascontiguousarray(v, dtype=np.float64).reshape(-1)


def matrix(m):
    if m is None:
        return None
    a = np.ascontiguousarray(m, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("expected a 2-D array")
    return a


def transform(eigenvalues, eigenvectors, mass, laplacian):
    """The native layer's transform arguments, in order: eigenvalues, eigenvectors, mass, laplacian."""
    return vector(eigenvalues), matrix(eigenvectors), sparse(mass), sparse(laplacian)


def field(x, name):
    """A real or complex 2-D array as float64 or complex128, C-ordered."""
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return np.ascontiguousarray(a, dtype=np.complex128 if np.iscomplexobj(a) else np.float64)


def stack(w, name):
    a = np.asarray(w)
    if a.ndim != 3:
        raise ValueError(f"{name} must be a 3-D array [members, rows, samples]")
    return np.ascontiguousarray(a, dtype=np.complex128)
