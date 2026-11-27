"""`nxr::select`: temporal events (the spans of a series where it is high, the bursts of an amplitude series, the switch-on and switch-off of an episode), and the levels of one manifold (levels.py)."""

from .events import Bursts, Spans, crossings, detectBursts, keepSpans, mad, median, mergeSpans
from .risefall import Fit, Stages, risefall, stagesOf
from .levels import Level, ProductLevel, dyadicLevel, kVertexLevel, levelComplex, lineLevel, productLevel, subjectTree, vertexLevel

__all__ = ["Bursts", "Fit", "Level", "ProductLevel", "dyadicLevel", "kVertexLevel", "levelComplex", "lineLevel", "productLevel", "subjectTree", "vertexLevel", "Spans", "Stages", "crossings", "detectBursts", "keepSpans", "mad", "median", "mergeSpans", "risefall", "stagesOf"]
