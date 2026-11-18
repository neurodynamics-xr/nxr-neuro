"""What there is: the pipelines `run` takes, the datasets of a datastore, the subjects of a dataset, what a subject holds (`nxr-neuro list ...`).

    nn.list.pipelines()                                   # each with its parameters' meaning, default and unit
    nn.list.datasets("study")
    nn.list.subjects("study/demo", where="age > 60")      # a catalogue dataset: the participants filter language
    nn.list.results("study/demo", "sub-01")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from . import _core

__all__ = ["pipelines", "datasets", "subjects", "results", "Pipeline"]


@dataclass
class Pipeline:
    """A pipeline `run` takes: its `name`, `form` ("definition"), `kind`, `scope`, `presets`, `summary` and `parameters` (each: name, default, meaning, unit, choices)."""

    name: str
    form: str
    summary: str
    parameters: list[dict[str, Any]] = field(default_factory=list)
    kind: str = ""
    scope: str = ""
    presets: list[str] = field(default_factory=list)

    def help(self) -> str:
        """The pipeline's help, made from its parameters' declared meaning, default and unit (what `nxr-neuro run NAME --help` prints)."""
        lines = [f"{self.name}: {self.summary}", ""]
        for p in self.parameters:
            default = "none" if p["default"] in (None, "") else p["default"]
            choices = f" (one of: {', '.join(map(str, p['choices']))})" if p.get("choices") else ""
            unit = "" if p.get("unit") in (None, "", "-") else f" [{p['unit']}]"
            lines.append(f"  {p['name']}{unit}: {p['meaning']}{choices}; default {default}")
        return "\n".join(lines)


def pipelines() -> list[Pipeline]:
    """The pipelines `nn.run` takes: the built-in `psd` and every shipped definition, by name."""
    return [Pipeline(**p) for p in json.loads(_core._pipelines())]


def datasets(datastore: str | os.PathLike) -> list[dict[str, Any]]:
    """The datasets of a datastore."""
    return json.loads(_core._datasets(os.fspath(datastore)))["datasets"]


def subjects(dataset: str | os.PathLike, where: str | None = None, names: list[str] | None = None) -> list[dict[str, Any]]:
    """The subjects of a dataset. In a catalogue dataset `where` (a filter over the participant variables, version 1 of the filter language) or `names` (by name or id)
    selects; not both."""
    return json.loads(_core._subjects(os.fspath(dataset), list(names or []), where))["subjects"]


def results(dataset: str | os.PathLike, subject: str, kind: str = "results") -> list[dict[str, Any]]:
    """What a subject holds, by `kind`: "results" (every complete field and statistic a computation stored), "recordings" or "kernels": a list of name, kind, table."""
    return json.loads(_core._results(os.fspath(dataset), subject, kind))
