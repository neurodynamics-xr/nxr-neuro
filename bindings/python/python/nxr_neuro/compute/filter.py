"""Zero-phase filters of a record (`nxr/filter.h`): Brainstorm's FIR (Kaiser window) and IIR (Butterworth notch and band-stop) filters as documented.

Each function filters a whole in-memory record [channels, samples]; the result has the same shape and precision. The options are the library's
`FilterOptions` (the same names as on the command line).
"""

from __future__ import annotations

from typing import Sequence

from .. import _core

_OPTIONS = """\
    sfreq : float
        Sampling frequency, Hz.
    method : {"fir", "iir"}, optional
        Unset: the reference's (FIR, but IIR for bandstop and notch).
    ripple : float
        Kaiser: the passband and stopband deviation (linear); the stopband attenuation is -20 log10(ripple) = 60 dB for the default 1e-3
        (Brainstorm "relax": 1e-2).
    transition_bandwidth : float, optional
        Hz, each transition band (the lower edge's with firwin). Unset: the reference's rule.
    edge : {"mean_of_transient", "zeros"}
        FIR: the two ends of the record are padded with the mean of the first and last K samples (Brainstorm; default), or with nothing.
    transient_energy : float
        mean_of_transient: K = the shortest one-sided impulse response holding this fraction of its energy (default 0.99).
    transient_samples : int, optional
        mean_of_transient: K stated (samples), instead of the energy rule.
    remove_mean : bool
        Subtract the record's mean before filtering; added back only when the filter has no high-pass part (Brainstorm; default True).
    notch_bandwidth : float, optional
        Hz: Brainstorm the 3-dB bandwidth of each notch (default 1 Hz), MNE the width of each stop band (default f / 200).
    bandstop_order : int
        Butterworth prototype order N (the filter has 2N poles; Brainstorm: fieldtrip's 'but', N = 4)."""


def _run(kind, x, sfreq, lower, upper, frequencies, method, ripple, transition_bandwidth, edge, transient_energy, transient_samples, remove_mean,
         notch_bandwidth, bandstop_order):
    return _core.filter(
        kind, x, float(sfreq), float(lower), float(upper), [float(f) for f in frequencies], method, float(ripple),
        None if transition_bandwidth is None else float(transition_bandwidth), edge, float(transient_energy),
        None if transient_samples is None else int(transient_samples), bool(remove_mean),
        None if notch_bandwidth is None else float(notch_bandwidth), int(bandstop_order),
    )


def lowpass(x, upper: float, sfreq: float, method=None, ripple=1e-3, transition_bandwidth=None, edge="mean_of_transient", transient_energy=0.99,
            transient_samples=None, remove_mean=True, notch_bandwidth=None, bandstop_order=4):
    """Low-pass filter (`nxr::filter::lowpass`).

    Parameters
    ----------
    x : array [channels, samples]
        The record, float32 or float64.
    upper : float
        Hz: the lowpass edge.
""" + _OPTIONS + """
    Returns
    -------
    array [channels, samples], in the input's precision.
    """
    return _run("lowpass", x, sfreq, 0.0, upper, (), method, ripple, transition_bandwidth, edge, transient_energy, transient_samples, remove_mean,
                notch_bandwidth, bandstop_order)


def highpass(x, lower: float, sfreq: float, method=None, ripple=1e-3, transition_bandwidth=None, edge="mean_of_transient", transient_energy=0.99,
             transient_samples=None, remove_mean=True, notch_bandwidth=None, bandstop_order=4):
    """High-pass filter (`nxr::filter::highpass`).

    Parameters
    ----------
    x : array [channels, samples]
        The record, float32 or float64.
    lower : float
        Hz: the highpass edge.
""" + _OPTIONS + """
    Returns
    -------
    array [channels, samples], in the input's precision.
    """
    return _run("highpass", x, sfreq, lower, 0.0, (), method, ripple, transition_bandwidth, edge, transient_energy, transient_samples, remove_mean,
                notch_bandwidth, bandstop_order)


def bandpass(x, lower: float, upper: float, sfreq: float, method=None, ripple=1e-3, transition_bandwidth=None, edge="mean_of_transient",
             transient_energy=0.99, transient_samples=None, remove_mean=True, notch_bandwidth=None, bandstop_order=4):
    """Band-pass filter (`nxr::filter::bandpass`).

    Parameters
    ----------
    x : array [channels, samples]
        The record, float32 or float64.
    lower, upper : float
        Hz: the lower and upper passband edges (0 < lower < upper < Nyquist).
""" + _OPTIONS + """
    Returns
    -------
    array [channels, samples], in the input's precision.
    """
    return _run("bandpass", x, sfreq, lower, upper, (), method, ripple, transition_bandwidth, edge, transient_energy, transient_samples, remove_mean,
                notch_bandwidth, bandstop_order)


def bandstop(x, lower: float, upper: float, sfreq: float, method=None, ripple=1e-3, transition_bandwidth=None, edge="mean_of_transient",
             transient_energy=0.99, transient_samples=None, remove_mean=True, notch_bandwidth=None, bandstop_order=4):
    """Band-stop filter (`nxr::filter::bandstop`).

    Parameters
    ----------
    x : array [channels, samples]
        The record, float32 or float64.
    lower, upper : float
        Hz: the lower and upper stopband edges.
""" + _OPTIONS + """
    Returns
    -------
    array [channels, samples], in the input's precision.
    """
    return _run("bandstop", x, sfreq, lower, upper, (), method, ripple, transition_bandwidth, edge, transient_energy, transient_samples, remove_mean,
                notch_bandwidth, bandstop_order)


def notch(x, frequencies: Sequence[float], sfreq: float, method=None, ripple=1e-3, transition_bandwidth=None, edge="mean_of_transient",
          transient_energy=0.99, transient_samples=None, remove_mean=True, notch_bandwidth=None, bandstop_order=4):
    """Notch filter at the line frequency and its harmonics (`nxr::filter::notch`).

    Parameters
    ----------
    x : array [channels, samples]
        The record, float32 or float64.
    frequencies : sequence of float
        Hz: the line frequency and its harmonics.
""" + _OPTIONS + """
    Returns
    -------
    array [channels, samples], in the input's precision.
    """
    return _run("notch", x, sfreq, 0.0, 0.0, frequencies, method, ripple, transition_bandwidth, edge, transient_energy, transient_samples, remove_mean,
                notch_bandwidth, bandstop_order)
