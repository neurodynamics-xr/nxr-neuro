"""Temporal events (`nxr::select`): the spans of a series where it is high, the bursts of an amplitude series, and the switch-on and switch-off of an episode."""

from .events import Bursts, Spans, crossings, detectBursts, keepSpans, mad, median, mergeSpans
from .risefall import Fit, Stages, risefall, stagesOf

__all__ = ["Bursts", "Fit", "Spans", "Stages", "crossings", "detectBursts", "keepSpans", "mad", "median", "mergeSpans", "risefall", "stagesOf"]
