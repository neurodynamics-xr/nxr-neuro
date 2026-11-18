"""Spans of a series and the bursts of an amplitude series (`nxr/select/events.h`).

A span is [start, stop) in samples of the series given (0-based, stop exclusive), with the sample and value of the extreme inside. The series is ANY series
(a sensor channel, a source or region signal, an amplitude): the functions never know which. Span-wide thresholds (a median, a percentile) are the caller's.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ... import _core


class Spans(NamedTuple):
    start: np.ndarray    # int64 [n], first sample
    stop: np.ndarray     # int64 [n], one past the last sample
    peak: np.ndarray     # int64 [n], the sample of the extreme value inside (the maximum for crossings)
    value: np.ndarray    # float64 [n], that value


class Bursts(NamedTuple):
    spans: Spans
    median: float        # the baseline used (the caller's, or the median over the finite values)
    sd: float            # its robust SD (1.4826 * MAD, or the caller's)
    enter: float         # the level to enter, in the series' units
    exit: float          # the level to extend, in the series' units
    good: int            # finite values in the series
    gap: int             # the durations as compared, in samples: spans closer than this are joined
    min_length: int      # bursts shorter than this are dropped
    max_length: int      # and longer than this (INT64_MAX when unbounded)
    skipped: str         # empty, or why nothing was detected


def _series(x, name="x") -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be a 1-D series")
    return a


def _spans(spans) -> Spans:
    s = Spans(*(spans if not hasattr(spans, "start") else (spans.start, spans.stop, spans.peak, spans.value)))
    return Spans(np.ascontiguousarray(s.start, dtype=np.int64), np.ascontiguousarray(s.stop, dtype=np.int64),
                 np.ascontiguousarray(s.peak, dtype=np.int64), np.ascontiguousarray(s.value, dtype=np.float64))


def median(x) -> float:
    """The median of a 1-D series (`nxr::select::median`)."""
    return _core.select_median(_series(x))


def mad(x) -> float:
    """The robust SD of a 1-D series: 1.4826 * median |x - median| (`nxr::select::mad`)."""
    return _core.select_mad(_series(x))


def crossings(x, entry: float, exit: float) -> Spans:
    """The spans where x >= `exit`, each holding at least one sample >= `entry` (`nxr::select::crossings`).

    Hysteresis: a span is entered high and extended while above the low level; entry = exit is a plain threshold. `peak` and `value` are the maximum inside.
    A non-finite sample ends a span. Lengths and positions are in samples of x.
    """
    return Spans(*_core.select_crossings(_series(x), float(entry), float(exit)))


def merge_spans(spans, gap: int) -> Spans:
    """The spans whose gap (next start - previous stop) is below `gap` samples joined (`nxr::select::mergeSpans`).

    spans : a `Spans`, or any object with start, stop, peak and value arrays.
    """
    s = _spans(spans)
    return Spans(*_core.select_merge_spans(s.start, s.stop, s.peak, s.value, int(gap)))


def keep_spans(spans, min_length: int = 0, max_length: int | None = None) -> Spans:
    """The spans whose length (stop - start, in samples) is in [min_length, max_length] kept (`nxr::select::keepSpans`); max_length None: no upper bound."""
    s = _spans(spans)
    return Spans(*_core.select_keep_spans(s.start, s.stop, s.peak, s.value, int(min_length), None if max_length is None else int(max_length)))


def detect_bursts(
    amplitude,
    step: float,
    enter_sd: float = 1.5,
    exit_sd: float = 1.0,
    merge_s: float = 0.1,
    min_s: float = 0.4,
    max_s: float = math.inf,
    baseline=None,
) -> Bursts:
    """The bursts of an amplitude series (`nxr::select::detectBursts`; Rheome's labelalpha in envelope mode, nxr's burst_stages / spindle_envelope).

    The robust baseline (the median and 1.4826 * MAD over the finite values, or the caller's), enter at median + enter_sd * SD, extend while >= median + exit_sd * SD,
    gaps joined, spans kept by length. The series is ANY amplitude (a sensor's envelope per sample, a tile measurement per tile, a region's mean).

    Parameters
    ----------
    amplitude : array [n]
        The series. A non-finite value (a bad sample or tile) ends a span and is left out of the baseline.
    step : float
        Seconds between its samples (1 / sfreq per sample, a tile's width per tile).
    enter_sd, exit_sd : float
        Robust SDs above the median to enter (default 1.5) and to extend (default 1.0).
    merge_s : float
        s: gaps lasting up to this are joined (default 0.1). Compared in whole samples.
    min_s, max_s : float
        s: the bursts kept last at least min_s (default 0.4) and at most max_s (default: no bound).
    baseline : (median, sd), optional
        The caller's baseline, in the series' units (a span-wide measurement); default: computed over the series' finite values.

    Returns
    -------
    Bursts
        spans (0-based, stop exclusive, in samples of the series) with the baseline and levels used, the durations as compared in samples, and `skipped`
        (why nothing was detected: fewer than 2 good samples, no spread).
    """
    b = None if baseline is None else (float(baseline[0]), float(baseline[1]))
    r = _core.select_detect_bursts(_series(amplitude, "amplitude"), float(step), float(enter_sd), float(exit_sd), float(merge_s), float(min_s), float(max_s), b)
    return Bursts(Spans(*r[0]), *r[1:])
