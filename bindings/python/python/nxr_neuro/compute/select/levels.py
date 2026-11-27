"""The levels of one manifold (`nxr/select/levels.h`; spec docs/specs/2026-11-26-levels-of-one-manifold.md).

A level is a level of a stored grid, never a manifold of its own: it is computed from the fine complex (the operator mesh's DEC, `operators.assembleDECOperators`,
and the vertices' areas) and the grid's codes (`subjectTree`, or a stored grid's), and it is navigable as a cell complex: `Level.neighbours`, `twin`, `next`, `parent`,
`children`, `area`, `star`, `d0`, `d1`, `crossing`, `laplacian`, and between levels `reductionOperator`, `expansionOperator`, `reduce` (by merge rule). A time line's
levels are closed form (`lineLevel`). `productLevel(space, time)` is the separable product, addressed ((surface, l, code), (time, k, code)).
"""

from __future__ import annotations

import numpy as np

from ... import _core

Level = _core.Level
ProductLevel = _core.ProductLevel
kVertexLevel = _core.select_kVertexLevel


def _ints(a):
    return np.ascontiguousarray(a, dtype=np.int32)


def _vec(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())


def levelComplex(d0, star0, star1, cell_of, cells: int, d1=None) -> Level:
    """The quotient of a fine complex (d0 [E x V] sparse, star0 [V], star1 [E], optional d1 [F x E]) by a vertex partition (`cell_of` [V], -1: none) into `cells` cells."""
    return _core.select_levelComplex(d0, d1, _vec(star0), _vec(star1), _ints(cell_of), int(cells))


def vertexLevel(d0, star0, star1, d1=None, codes=None, depth: int = 0, slots: int = 0) -> Level:
    """The fine complex as a level (each vertex its own cell); with a dyadic grid's `codes` (at `depth`, `slots` piece slots) its vertices' parents are the grid's finest tiles."""
    return _core.select_vertexLevel(d0, d1, _vec(star0), _vec(star1), None if codes is None else _ints(codes), int(depth), int(slots))


def dyadicLevel(d0, star0, star1, codes, depth: int, slots: int, level: int, d1=None) -> Level:
    """Level `level` (0 .. depth) of the dyadic grid whose codes at `depth` are `codes` (h 2^depth + t, -1: none): the vertex's cell is code >> (depth - level)."""
    return _core.select_dyadicLevel(d0, d1, _vec(star0), _vec(star1), _ints(codes), int(depth), int(slots), int(level))


def lineLevel(samples: int, rate: float, k: int) -> Level:
    """Level k of a time line of `samples` samples at `rate` Hz: tiles of 2^k samples from the first, the last as long as the line (closed form)."""
    return _core.select_lineLevel(int(samples), float(rate), int(k))


def productLevel(space: Level, time: Level, coupling: dict | None = None) -> ProductLevel:
    """The product of a surface level and a time level, separable. `coupling={"kind": "dispersion", "speed": c}` is the typed slot of a later step (refused now)."""
    return _core.select_productLevel(space, time, coupling)


def subjectTree(vertices, faces, depth: int = -1) -> dict:
    """The dyadic grid of a mesh as `select_dyadic_surface --rule subject` makes it: `codes` (h 2^depth + t per vertex), `depth`, `finest`, `slots`, `diameter` per level."""
    return _core.select_subjectTree(np.ascontiguousarray(vertices, dtype=np.float64), np.ascontiguousarray(faces, dtype=np.int32), int(depth))
