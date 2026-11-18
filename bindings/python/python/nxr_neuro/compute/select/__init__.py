"""Temporal events (`nxr::select`): the spans of a series where it is high, the bursts of an amplitude series, and the switch-on and switch-off of an episode."""

from .events import Bursts, Spans, crossings, detect_bursts, keep_spans, mad, median, merge_spans
from .risefall import Fit, Stages, risefall, stages

__all__ = ["Bursts", "Fit", "Spans", "Stages", "crossings", "detect_bursts", "keep_spans", "mad", "median", "merge_spans", "risefall", "stages"]
