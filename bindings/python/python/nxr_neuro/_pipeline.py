"""The pipeline level: `run` a pipeline on a subject of a dataset, and `read` what is stored.

`run` is `nxr-neuro run`: the same library calls, so a result made here is the same cached result the command line makes (and finds). The time range is always the
caller's: `start` and `stop` in seconds on the recording's clock, or `all=True` (the whole recording, still read one chunk at a time); a call without one is refused.

    import nxr_neuro as nn
    r = nn.run("psd", "study/demo", "sub-01", recording="rest", set="brainstorm", segment_seconds=2.0, all=True)
    r.cache                       # "computed" the first time, "hit" the next
    psd = r.read()                # the stored statistic: psd.values [channels, frequencies], psd.meta
    nn.read("study/demo", "sub-01", "rest", start=0, stop=1200, rows=[0, 1]).values   # a recording's samples [0, 1200) of two channels, through the chunk service
    nn.read("study/demo", "sub-01", "sensors").positions                               # a channel manifold's positions; family="type": a labelling of it
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import _core

__all__ = ["run", "read", "RunResult", "Stored"]


@dataclass
class Stored:
    """A stored field, statistic, operator or selection (or a selection's measurement table) with its metadata.

    `arrays` holds its arrays by name (`values` for a field, statistic or operator; a sparse operator: `data`, `indices`, `indptr`; a measurement table: `code`, `value`,
    `weight`, `members`); `meta` what the node records about itself (class, name, id, kind, unit, and a statistic's measure, method, axes, options and provenance).
    An array is also an attribute: `stored.values`.
    """

    name: str
    meta: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def __getattr__(self, key: str):
        arrays = self.__dict__.get("arrays", {})
        if key in arrays:
            return arrays[key]
        raise AttributeError(key)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]


@dataclass
class RunResult:
    """What a run stored: `stored` (each output's `output`, `name`, `id`, `path`), `cache` ("hit" when everything was found stored, else "computed"), the library's own
    report of each run in `runs`, and `seconds`."""

    pipeline: str
    subject: str
    cache: str
    stored: list[dict[str, Any]]
    runs: list[dict[str, Any]]
    seconds: float
    dataset: str = ""

    def read(self, name: str | None = None) -> Stored:
        """A stored output read back (the first when `name` is not given)."""
        if not self.stored:
            raise ValueError("the run stored nothing")
        pick = self.stored[0]["name"] if name is None else name
        return read(self.dataset, self.subject, pick)


def run(
    pipeline: str,
    dataset: str | os.PathLike,
    subject: str,
    *,
    start: float | None = None,
    stop: float | None = None,
    all: bool = False,
    replace: bool = False,
    progress: Callable[[dict], None] | None = None,
    **parameters,
) -> RunResult:
    """Run `pipeline` on `subject` of `dataset` (a dataset folder), as `nxr-neuro run PIPELINE --dataset D --subject S` does.

    The time range is REQUIRED and the caller's: `start` and `stop` (seconds; both) or `all=True`; none is refused (`InvalidArgumentError`). (The command line's `--from` and
    `--to`; `from` is a Python keyword.) The pipeline's own parameters are keyword arguments (`nn.list.pipelines()` lists each one's meaning, default and unit; `psd`'s options are
    `recording`, `name`, `set`, `segment_seconds`, ...). `replace=True` moves another job's outputs of the same names aside (never deleted) and replaces them; without it they
    are refused. `progress` is called with each progress line (a dict: the command line's JSON lines). A rerun with the same pipeline, parameters, inputs and range is a cache
    hit: nothing is computed.

    Returns a `RunResult`: what was stored (names, ids, paths) and whether it was a cache hit.
    """
    text = _core._run(os.fspath(dataset), subject, pipeline, json.dumps(parameters, default=_plain), start, stop, bool(all), bool(replace),
                      None if progress is None else (lambda line: progress(json.loads(line))))
    r = json.loads(text)
    return RunResult(dataset=os.fspath(dataset), **r)


def read(
    dataset: str | os.PathLike,
    subject: str,
    name: str,
    *,
    measure: str | None = None,
    of: str | None = None,
    start: int | None = None,
    stop: int | None = None,
    rows: Sequence[int] | None = None,
    projectors: str = "active",
    reading: str = "samples",
    family: str | None = None,
) -> Stored:
    """The stored `name` of `subject` (a statistic, field, operator, selection or manifold, by name or id): its arrays and metadata. Nothing is written.

    - A statistic, an operator, a field with no time line: read whole (`values`).
    - A TIME-RESOLVED FIELD (a recording, a source series) is read through the chunk service, one chunk at a time: `start` and `stop` (SAMPLES, both or neither: neither reads
      it whole) and `rows` (its channels, in the order given; none: all) say what; `values` is [rows, samples] in its stored precision. `projectors` "active" (default: the
      recording's active projectors applied, as every run reads it) or "none". `reading="shape"` gives the metadata only, no sample: `meta` holds `rate` (Hz), `origin` (s),
      `n_samples`, `n_channels`, `channels` (the channel manifold's name), `unit`, `span`, `projection` (what was applied) and `producer`.
    - A selection with an array: `values`. A selection of SPANS (events, bad segments): `start`, `stop` (seconds on its line's clock), `code` (-1: none), and
      `meta["dictionary"]` [{code, name}]. With `measure`: its measurement table (`code`, `value`, `weight`, `members`); `of` names the field when several are measured.
    - A MANIFOLD: `positions` [n, 3] as stored (`meta["coordinate_unit"]` says their unit), `cells_<rank>` when it has cells; `meta` has `type`, `n`, `axes`, `names` (one name per
      element, when stored) and the `selections` on it. With `family` (e.g. "type", "unit", "quality", "hemisphere", "lobe"): that labelling of the manifold, `codes` (one per
      element, -1: none) and `meta["dictionary"]` [{code, name}].
    """
    meta_text, arrays = _core._read(os.fspath(dataset), subject, name, measure, of, None if start is None else int(start), None if stop is None else int(stop),
                                    [] if rows is None else [int(r) for r in rows], projectors, reading, family)
    meta = json.loads(meta_text)
    out = {}
    for key, dtype, shape, raw in arrays:
        out[key] = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<")).reshape(shape).astype(np.dtype(dtype), copy=True)
    return Stored(name=meta.get("name", name), meta=meta, arrays=out)


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{type(value).__name__} cannot be a pipeline parameter")
