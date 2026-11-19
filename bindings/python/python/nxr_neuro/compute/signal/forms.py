"""Forms on the intrinsic complex (`nxr/signal/intrinsic_forms.h`, `nxr::signal`): 1-forms <-> tangent vectors (sharp, flat), the phase 1-forms of a
complex field (phaseCurrent, phaseOneForm), and the Hodge–Dirac modes' reading of forms (modesOf, projectOneForm / synthesizeOneForm, projectTwoForm /
synthesizeTwoForm, exactPotential, coexactStream, splitPhase).

Every function takes the surface dict made by `geometry.surfaceArrays` and works on its intrinsic edges and faces (`geometry.complex`); the modes must be
solved on the same surface (`operators.hodgedirac.assemble(surface)` with its default options), or they are refused.
"""

from __future__ import annotations

import numpy as np

from ... import _core
from ..geometry import _contiguous


def sharp(surface: dict, w) -> np.ndarray:
    """A 1-form on the intrinsic edges -> a tangent vector at each vertex, in R^3.

    At each vertex the least-squares tangent vector v with v . t_e = w_e / L_e over its edges, t_e the edge's direction at the vertex (its signpost
    angle placed between the input edges that bracket it). Edges as `geometry.complex` gives them (tail -> tip).

    Parameters
    ----------
    w : array [nE] or [nE, m]
        The 1-form (m of them as columns).

    Returns
    -------
    array [nV, 3] or [nV, 3, m]
    """
    w = np.asarray(w, dtype=np.float64)
    single = w.ndim == 1
    J = _core.signal_sharp(_contiguous(surface), w.reshape(len(w), -1))
    J = np.asarray(J).reshape(-1, 3, J.shape[1])
    return J[:, :, 0] if single else J


def flat(surface: dict, J) -> np.ndarray:
    """A vertex vector field in R^3 -> a 1-form on the intrinsic edges: w_e = 1/2 L_e (J_tail . t_tail + J_tip . t_tip) (the field integrated along the
    edge, trapezoid rule; t_e the edge's direction at each end, tail -> tip).

    Parameters
    ----------
    J : array [nV, 3] or [nV, 3, m]

    Returns
    -------
    array [nE] or [nE, m]
    """
    J = np.asarray(J, dtype=np.float64)
    single = J.ndim == 2
    if J.shape[1] != 3:
        raise ValueError("the vector field is [n_vertices, 3] or [n_vertices, 3, m]")
    w = _core.signal_flat(_contiguous(surface), J.reshape(J.shape[0] * 3, -1))
    return np.asarray(w)[:, 0] if single else np.asarray(w)


def phaseCurrent(surface: dict, z) -> np.ndarray:
    """The phase current of a complex vertex field: J_e = Im(conj(z_tail) z_tip) = |z_tail| |z_tip| sin(arg z_tip - arg z_tail), a 1-form on the
    intrinsic edges (tail -> tip).

    Parameters
    ----------
    z : complex array [nV] or [nV, m]
        One column a frame (pointwise in time).

    Returns
    -------
    array [nE] or [nE, m]
    """
    z = np.asarray(z, dtype=np.complex128)
    J = _core.signal_phaseCurrent(_contiguous(surface), z.reshape(len(z), -1))
    return J[:, 0] if z.ndim == 1 else J


def phaseOneForm(surface: dict, x) -> np.ndarray:
    """The phase step of a complex vertex field along each intrinsic edge: w_e = arg(x_tip conj(x_tail)) (rad, in (-pi, pi]), [nE]."""
    return _core.signal_phaseOneForm(_contiguous(surface), np.asarray(x, dtype=np.complex128).ravel())[:, 0]


def modesOf(modes: dict, pencil: dict) -> dict:
    """The modes of a Hodge–Dirac solve (`operators.hodgeDiracModes(pencil, ...)`, without both signs) as the forms read them: ``P`` [nV, K] the
    0-form parts psi / sqrt 2, ``X2`` [nF, K] the 2-form parts chi / sqrt 2, ``mu`` [K], ``kind`` [K] (0 exact, 2 coexact)."""
    return _core.signal_modesOf(np.asarray(modes["eigenvectors"], dtype=np.float64), np.asarray(modes["eigenvalues"], dtype=np.float64),
                                np.asarray(modes["kind"], dtype=np.int32), int(pencil["nV"]), int(pencil["nE"]), int(pencil["nF"]))


def _columns(a):
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(len(a), -1), a.ndim == 1


def _shaped(out, single):
    return out[:, 0] if single else out


def projectOneForm(surface: dict, modes: dict, w) -> np.ndarray:
    """A 1-form on the intrinsic edges -> its coefficients in the modes' 1-form parts phi_k (exact: d psi / (mu sqrt 2); coexact:
    *1^-1 d1' *2 chi / (mu sqrt 2)): c_k = 2 <w, phi_k>_*1, inner products, no solve. w [nE] or [nE, m] -> [K] or [K, m]."""
    a, single = _columns(w)
    return _shaped(_core.signal_projectOneForm(_contiguous(surface), modes, a), single)


def synthesizeOneForm(surface: dict, modes: dict, coef) -> np.ndarray:
    """Coefficients -> the 1-form sum_k c_k phi_k on the intrinsic edges. Zero the coexact rows for the curl-free part, the exact rows for the
    divergence-free part, the rows with mu above a cut for a band-limited form. [K] or [K, m] -> [nE] or [nE, m]."""
    a, single = _columns(coef)
    return _shaped(_core.signal_synthesizeOneForm(_contiguous(surface), modes, a), single)


def projectTwoForm(surface: dict, modes: dict, sigma) -> np.ndarray:
    """A 2-form on the intrinsic faces -> its coefficients in the coexact modes' 2-form parts (0 for the exact modes). [nF(, m)] -> [K(, m)]."""
    a, single = _columns(sigma)
    return _shaped(_core.signal_projectTwoForm(_contiguous(surface), modes, a), single)


def synthesizeTwoForm(surface: dict, modes: dict, coef) -> np.ndarray:
    """Coefficients -> the 2-form sum over the coexact modes of c_k chi_k / sqrt 2. [K(, m)] -> [nF(, m)]."""
    a, single = _columns(coef)
    return _shaped(_core.signal_synthesizeTwoForm(_contiguous(surface), modes, a), single)


def exactPotential(modes: dict, coef) -> np.ndarray:
    """The exact part's potential: Phi = sum_exact c_k P_k / mu_k at the vertices, whose d0 is exactly the exact part of synthesizeOneForm(coef). For a
    phase 1-form, a timing map (rad; time = Phi / omega). [K(, m)] -> [nV(, m)]."""
    a, single = _columns(coef)
    return _shaped(_core.signal_exactPotential(modes, a), single)


def coexactStream(surface: dict, modes: dict, coef) -> np.ndarray:
    """The coexact part's stream function: the 2-form potential sum_coexact c_k X2_k / mu_k, read at the vertices (each vertex the area-weighted mean of
    its faces). Dirichlet (zero) on an open boundary. [K(, m)] -> [nV(, m)]."""
    a, single = _columns(coef)
    return _shaped(_core.signal_coexactStream(_contiguous(surface), modes, a), single)


def splitPhase(surface: dict, modes: dict, w) -> dict:
    """The split of a phase 1-form w [nE] by the modes: ``coef`` [K]; ``exact``, ``coexact`` [nE] (the timing and rotation parts); ``shareTiming``,
    ``shareRotation``, ``shareResidual`` (shares of the *1 energy); ``potential`` [nV] (exactPotential), ``stream`` [nV] (coexactStream),
    ``propagation`` [nV, 3] (-grad potential, sharpened), ``inconsistency`` [nV] (per vertex, the share of *1 |w|^2 its edges' exact part leaves)."""
    out = _core.signal_splitPhase(_contiguous(surface), modes, np.asarray(w, dtype=np.float64).ravel())
    out["propagation"] = np.asarray(out["propagation"]).reshape(-1, 3)
    return out
