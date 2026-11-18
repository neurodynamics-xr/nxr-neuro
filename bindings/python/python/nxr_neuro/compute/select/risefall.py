"""Where an amplitude episode switches on and off (`nxr/select/risefall.h`): the double-sigmoid fit of Rheome's `+rheome/+detect/risefall.m`.

    a(t) = b + h S((t - t_on)/tau_rise) S(-(t - t_off)/tau_fall)                      model "linear"      S(x) = 1/(1 + e^-x)
    a(t) = sqrt(b^2 + (h S((t - t_on)/tau_rise) S(-(t - t_off)/tau_fall))^2)          model "quadrature"  an episode on a background adds in power

fitted by bounded nonlinear least squares to the series, smoothed by a symmetric moving mean and scaled to unit maximum, over [t0 - margin, t1 + margin].
t_on and t_off are the HALF-rise and HALF-fall times. What it cannot show: the fit is of the amplitude's shape, not of its sources.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ... import _core


class Fit(NamedTuple):
    t_on: float             # s, on the series' clock (sample i is at i / sfreq)
    t_off: float
    tau_rise: float         # s
    tau_fall: float         # s
    baseline: float         # b, the amplitude's units
    height: float           # h
    r2: float               # of the fit on the unit scale (scale-free)
    ok: bool                # False: a window under 11 samples, a series of no amplitude, or t_off <= t_on (every number NaN)
    at_bound: bool          # a rise or fall time within 1 % of tau_max
    buildup: tuple          # (start, end) s: [t_on - 2 tau_rise, t_on + 2 tau_rise]
    plateau: tuple          # (start, end) s: [t_on + 2 tau_rise, t_off - 2 tau_fall]; NaN when they cross
    decay: tuple            # (start, end) s: [t_off - 2 tau_fall, t_off + 2 tau_fall]
    has_plateau: bool
    kept: bool | None       # the fit gate (ok, r2 >= min_r2, not at a bound) when min_r2 is given, else None


class Stages(NamedTuple):
    onset: tuple            # (start, end) s: [t_on - 2 tau_rise, t_on + 2 tau_rise]
    within: tuple           # (start, end) s: the plateau; NaN when the onset and offset meet
    offset: tuple           # (start, end) s: [t_off - 2 tau_fall, t_off + 2 tau_fall]
    has_within: bool
    valid: bool             # the four numbers were finite and the rise and fall times above 0


def risefall(
    amplitude,
    sfreq: float,
    t0: float,
    t1: float,
    model: str = "linear",
    margin: float = 1.0,
    smooth: float = 0.1,
    tau_max: float = 1.0,
    min_r2: float | None = None,
) -> Fit:
    """Fit the switch-on and switch-off of the episode [t0, t1] s of an amplitude series (`nxr::select::fit`).

    Parameters
    ----------
    amplitude : array [n]
        Any amplitude series (a region's mean, the maximum over tiles).
    sfreq : float
        Samples per second.
    t0, t1 : float
        The candidate episode, s.
    model : {"linear", "quadrature"}
        The model above (default "linear"). Rheome's "posterior" is not built and is refused.
    margin : float
        s either side of [t0, t1] included in the fit (default 1.0).
    smooth : float
        s: the symmetric moving mean applied first (default 0.1).
    tau_max : float
        s: the upper bound of the rise and fall times (default 1.0).
    min_r2 : float, optional
        When given, `kept` is the fit gate (`nxr::select::passesGate`): the fit is ok, r2 >= min_r2 and the times are not at their bound.
    """
    a = np.ascontiguousarray(amplitude, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("amplitude must be a 1-D series")
    return Fit(*_core.select_risefall(a, float(sfreq), float(t0), float(t1), model, float(margin), float(smooth), float(tau_max),
                                      None if min_r2 is None else float(min_r2)))


def stages(t_on: float, t_off: float, tau_rise: float, tau_fall: float) -> Stages:
    """The stages of an episode from a fit's four numbers, s (`nxr::select::stagesOf`): onset, within (the plateau) and offset. NaN in: no stage."""
    return Stages(*_core.select_stages(float(t_on), float(t_off), float(tau_rise), float(tau_fall)))
