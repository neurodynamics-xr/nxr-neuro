"""The command line's commands as functions: `nxr-neuro <verb> <noun> --option VALUE` is `nxr_neuro.<verb>.<noun>(option=VALUE)`.

The rule (docs/specs/2026-11-19-io-service.md, section 2): the same verbs, nouns and option names, dashes as underscores (`--from-sphere` is `from_sphere`); a Python keyword
takes a trailing underscore (the verb `import` is the module `nxr_neuro.import_`, the option `--from` is `from_`). The positional argument of a command (`import subject
PROTOCOL`, `read dataset FOLDER`) is the function's one positional parameter; every option is given by name. A flag is a boolean, a repeatable option a list. Required options
are required; an option left out takes the command line's default (the library's request is the same one the command line fills).

What comes back is the command's result object (its `"stage": "result"` line) as a `Result`, a dict; where the command line writes an array to a file or prints its
checksum, the array is in `Result.arrays` (NumPy, in its stored precision). Arrays go in as NumPy arrays (or any array-like, or a path to a .npy file); they are handed to the
library in memory (dtype, shape, bytes), under the name the producer records for them (`values.npy`); no temporary file is written.

COMMANDS is the table this module is made from; `tests/test_commands.py` checks it against the command line's table (cli/command_table.cpp).
"""

from __future__ import annotations

import builtins
import inspect
import json
import keyword
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import _core
from ._errors import InvalidArgumentError

# (verb, noun, positional, options): the positional argument's name ("" none, a trailing ? optional); each option `name`, `*` required, `+` repeatable (a list), `!` a flag.
COMMANDS: list[tuple[str, str, str, str]] = [
    # datasets
    ("create", "dataset", "", "atlas_finest dataset datastore default lc_scales modes name source_path source_protocol source_tool store_format templates"),
    ("list", "datasets", "", "datastore*"),
    ("show", "dataset", "", "dataset*"),
    ("delete", "dataset", "", "datastore* name*"),
    ("recover", "dataset", "", "dataset*"),
    ("compact", "dataset", "", "dataset* dry_run! no_shard! no_small_files! subject"),
    ("split", "dataset", "", "dataset* dry_run! registered_by*"),
    ("join", "dataset", "", "dataset*"),
    ("migrate", "eigensolve", "", "dataset* name subject* templates+"),
    ("migrate", "glue", "", "dataset* hemispheres left name right subject* templates+"),
    ("migrate", "measurements", "", "dataset* dry_run! remove_statistics!"),
    ("migrate", "selections", "", "dataset* dry_run!"),
    # subject packages and the catalogue
    ("create", "recording", "", "acquisition amplitude channels duration frequency+ name noise package* seed sfreq synthetic!"),
    ("create", "package", "", "home_dataset* kind name* package*"),
    ("create", "index", "", "dataset* force!"),
    ("register", "package", "", "dataset* package+ registered_by* replace! scan! templates+"),
    ("list", "members", "", "dataset*"),
    ("verify", "package", "", "package*"),
    ("show", "package", "", "package*"),
    ("release", "template", "", "into* name* package* released_by* template_version*"),
    ("release", "redistributable", "", "into* package* released_by*"),
    ("create", "template", "", "accept_unlisted_sources! cortex_mask! scalp! hcp_end_surface hcp_points hcp_trk home_dataset max_end_distance name package* parcellation+ recipe* sources* surface+"),
    ("create", "registration", "", "attribution from* from_sphere* home_dataset licence method name* neighbours one_direction! package* redistributable source templates+ to* to_sphere*"),
    ("list", "templates", "", "templates+"),
    # subjects
    ("create", "subject", "", "dataset* kind name*"),
    ("list", "subjects", "", "dataset* subjects where"),
    ("show", "subject", "", "dataset* subject* templates+"),
    ("delete", "subject", "", "by dataset* node subject* templates+"),
    ("add", "surface", "file?", "by dataset* faces name no_primary! primary! scale subject* vertices templates+"),
    ("list", "recordings", "", "dataset* subject* templates+"),
    ("list", "kernels", "", "dataset* subject* templates+"),
    ("list", "results", "", "dataset* subject* templates+"),
    ("list", "manifolds", "", "after columns dataset* kind limit name on subject templates+"),
    ("list", "fields", "", "after columns dataset* kind limit name on subject templates+"),
    ("list", "selections", "", "after columns dataset* kind limit name on subject templates+"),
    ("list", "operators", "", "after columns dataset* kind limit name on subject templates+"),
    ("list", "statistics", "", "after columns dataset* kind limit name on subject templates+"),
    ("list", "dictionaries", "", "after columns dataset* kind limit name subject templates+"),
    ("list", "spaces", "", "after columns dataset* kind limit name subject templates+"),
    ("list", "acquisitions", "", "after columns dataset* limit name subject templates+"),
    ("list", "jobs", "", "after columns dataset* kind limit subject templates+"),
    ("read", "subject", "", "classes dataset* related! subject* templates+"),
    ("list", "pipelines", "", ""),
    # reading and querying
    ("read", "dataset", "folder", "array+ out region"),
    ("read", "crossspectrum", "", "dataset* name* pair reading seed subject* weights templates+"),
    ("read", "field", "", "all! budget dataset* from name* onto out projectors region rows shape_only! subject* to templates+"),
    ("read", "statistic", "", "budget dataset* name* out region subject* templates+"),
    ("read", "operator", "", "budget dataset* name* onto out region subject* templates+"),
    ("read", "selection", "", "budget dataset* field measure name* out region subject* templates+"),
    ("read", "manifold", "", "budget dataset* family name* out region subject* templates+"),
    ("query", "measurements", "", "band dataset* factor field limit max_value measure* min_value order selection* spatial_scale subject* subjects tile templates+ where"),
    ("compile", "", "definition", "dataset inputs set+ signatures subject"),
    # add <class>
    ("add", "field", "", "append! begin! by by_operator cell commit! contribution+ dataset* derived_from description dtype kind manifold name producer region replace! shape subject time unit value_type values write templates+"),
    ("add", "selection", "", "by dataset* description entries indices manifold* members name* of_field producer replace! spans subject* type* weights templates+"),
    ("add", "statistic", "", "append! axes+ begin! by commit! dataset* description dtype measure method name of_field partition region replace! shape subject unit value_type values write templates+"),
    ("add", "measurements", "", "by dataset* field measure*+ replace! selection* subject* unit templates+"),
    ("add", "operator", "", "by data dataset* description from_manifold* from_selection indices indptr kind layout method name* params producer replace! subject* to_manifold* values templates+"),
    ("add", "points", "", "by coordinate_system dataset* description name* positions* replace! subject* templates+"),
    ("add", "line", "", "by coordinates dataset* dimension kinds n name* origin replace! spacing subject* unit* templates+"),
    ("add", "product", "", "by dataset* factors* name* subject* templates+"),
    ("add", "dictionary", "", "by dataset* description entries* name* scope* subject templates+"),
    # edit
    ("edit", "manifold", "", "dataset* description label name* primary subject* templates+"),
    ("edit", "operator", "", "by dataset* name* note status* subject* templates+"),
    ("edit", "dictionary", "", "colour dataset* description entries entry label name* name_of params replace! subject templates+"),
    # import
    ("import", "list", "protocol", ""),
    ("import", "scan", "protocol", "out prefer summary!"),
    ("import", "subject", "protocol", "dataset* no_sensor_mesh! resample subject* templates+"),
    ("import", "template", "protocol", "dataset* surface+"),
    ("import", "group", "protocol", "dataset*"),
    ("import", "templates", "templates_dir", "dataset* subject surface trk"),
    ("import", "condition", "protocol", "all_surfaces! channel_file condition* dataset* extra_surface+ include_imported! no_gain! no_raw! recording resample subject* surface templates+"),
    ("import", "surface", "", "bst_root dataset* file+ subject* templates+"),
    ("import", "mri", "", "anat* dataset* only+ subject* templates+"),
    ("import", "covariance", "", "condition* dataset* kind name protocol* replace! subject* templates+"),
    ("import", "inverse", "", "acquisition* bst_root* channel_file channels_id dataset* file* forward_id name subject* surface_id templates+"),
    ("import", "map", "", "acquisition bst_root* dataset* file*+ provenance replace! subject* templates+"),
    ("import", "resample", "", "dataset* name recording* subject* to* templates+"),
    ("import", "badsegments", "", "dataset* from* replace! subject* templates+"),
    ("import", "events", "", "dataset* from* replace! subject* templates+"),
    ("import", "bids", "", "anonymize! dataset* from* in_place! no_participants! registered_by subject+"),
    ("import", "fif", "", "acquisition anonymize! attribution channels from* kind licence name package* recording source_root surface"),
    ("import", "nifti", "", "anonymize! attribution field_kind from* kind label_names licence name package* parcellation source_root unit"),
    ("import", "freesurfer", "", "annotations+ anonymize! attribution field_kind from* licence name orig overlays+ package* source_root surfaces+"),
    ("import", "gifti", "", "anonymize! attribution field_kind from* kind licence name package* parcellation source_root surface unit"),
    ("import", "cifti", "", "anonymize! attribution field_kind from* left licence name package* parcellation right source_root unit"),
    ("import", "transform", "", "channels from* name package*"),
    ("import", "recording", "", "acquisition anonymize! attribution format from* in_place! licence montage name package* source_root"),
    ("import", "participants", "", "dataset* sidecar tsv*"),
    # export
    ("export", "participants", "", "dataset* sidecar tsv*"),
    ("export", "measurements", "", "dataset* exclude fields* keep_all! labelling* layout out* recipe scaling statistics subjects table*"),
    ("export", "gifti", "", "all! field from hemispheres! kind labelling out* package* surface_unit to with_surface!"),
    ("export", "tsv", "", "field measures out* package* partition statistic"),
    ("export", "fif", "", "all! covariance field forward from inverse out* package* to"),
    ("export", "bids-derivatives", "", "all! from measurements out* package* results* run session space stc! task to with_surface!"),
]

# the options that are JSON (a dict or list given is written as JSON text; a string is the text, or for `entries` the path of a JSON file)
JSON_OPTIONS = {
    ("add", "field"): {"producer"},
    ("add", "selection"): {"producer"},
    ("add", "operator"): {"params", "producer"},
    ("add", "dictionary"): {"entries"},
    ("edit", "dictionary"): {"params", "entries"},
}

# the options that take an array: handed to the library in memory (or as a .npy path) (`add measurements --measure NAME=FILE.npy`: a dict of arrays)
ARRAY_OPTIONS = {
    ("add", "field"): {"values"},
    ("add", "selection"): {"members", "indices", "spans", "weights"},
    ("add", "statistic"): {"values"},
    ("add", "operator"): {"values", "data", "indices", "indptr"},
    ("add", "points"): {"positions"},
    ("add", "line"): {"coordinates", "kinds"},
    ("add", "surface"): {"vertices", "faces"},
}


def py_name(word: str) -> str:
    """The Python name of a command line word: dashes as underscores, a Python keyword with a trailing underscore."""
    word = word.replace("-", "_")
    return word + "_" if keyword.iskeyword(word) else word


@dataclass
class Option:
    name: str            # the command line's name with underscores ("from", "home_dataset")
    required: bool = False
    multi: bool = False
    flag: bool = False

    @property
    def python(self) -> str:
        return py_name(self.name)


def parse_options(text: str) -> list[Option]:
    out = []
    for w in text.split():
        o = Option(w.rstrip("*+!"))
        o.required, o.multi, o.flag = "*" in w, "+" in w, "!" in w
        out.append(o)
    return out


class Result(dict):
    """A command's result object (a dict), with its arrays by name in `.arrays` (NumPy, in the stored precision; empty when the command hands back none)."""

    def __init__(self, result: dict, arrays: dict[str, np.ndarray] | None = None):
        super().__init__(result)
        self.arrays: dict[str, np.ndarray] = arrays or {}

    def __repr__(self) -> str:
        extra = f", arrays={ {k: (v.dtype.name, v.shape) for k, v in self.arrays.items()} }" if self.arrays else ""
        return f"Result({dict.__repr__(self)}{extra})"


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not a value a command takes")


def _memory(option: str, name: str, value):
    """An array for the library in memory: (option, name recorded, dtype, shape, bytes), C order, little-endian, its own precision; a path is returned as it is."""
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    a = np.asarray(value)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{option}: an array of {a.dtype} is not taken (float32, float64, integers, bool)")
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    a = np.ascontiguousarray(a)
    return (option, f"{name}.npy", a.dtype.name, tuple(int(d) for d in a.shape), a.tobytes())


def _json_value(value):
    """`compile`'s signatures and inputs: a dict or list as it is, or a JSON file read."""
    if isinstance(value, (str, os.PathLike)):
        with builtins.open(value, encoding="utf-8") as f:
            return json.load(f)
    return value


def _arrays(raw) -> dict[str, np.ndarray]:
    out = {}
    for key, dtype, shape, data in raw:
        dt = np.dtype(dtype)
        out[key] = np.frombuffer(data, dtype=dt.newbyteorder("<")).reshape(shape).astype(dt, copy=True)
    return out


def call(verb: str, noun: str, options: dict[str, Any], progress: Callable[[dict], None] | None = None, handle=None) -> Result:
    """One command with its options (keyed by the command line's names, underscores for dashes): its Result."""
    array_options = ARRAY_OPTIONS.get((verb, noun), set())
    given = {}
    in_memory = []
    for k, v in options.items():
        if v is None:
            continue
        if k in array_options:
            m = _memory(k, k, v)
            if isinstance(m, str):
                v = m
            else:
                in_memory.append(m)
                continue
        elif (verb, noun, k) == ("add", "measurements", "measure") and isinstance(v, dict):
            paths = []
            for name, a in v.items():
                m = _memory(f"measure.{name}", name, a)
                if isinstance(m, str):
                    paths.append(f"{name}={m}")
                else:
                    in_memory.append(m)
            v = paths or None
            if v is None:
                continue
        elif k in JSON_OPTIONS.get((verb, noun), ()) and isinstance(v, (dict, list, tuple)):
            v = json.dumps(v, default=_plain)
        elif (verb, noun) == ("compile", "") and k in ("signatures", "inputs"):
            v = _json_value(v)
        elif (verb, noun, k) == ("compile", "", "set") and isinstance(v, dict):
            v = [f"{name}={json.dumps(_plain(x) if isinstance(x, (np.generic, np.ndarray)) else x)}" for name, x in v.items()]
        given[k] = v
    text, raw = _core._command(verb, noun, json.dumps(given, default=_plain), None if progress is None else (lambda line: progress(json.loads(line))), in_memory or None, handle)
    return Result(json.loads(text), _arrays(raw))


def _call_form(verb: str, noun: str, positional: str, options: list[Option]) -> str:
    args = ([positional.rstrip("?")] if positional else []) + [f"{o.python}=..." for o in options if o.required]
    where = f"{py_name(verb)}.{py_name(noun)}" if noun else py_name(verb)
    return f"nxr_neuro.{where}({', '.join(args)}{', ...' if any(not o.required for o in options) else ''})"


def make(verb: str, noun: str, positional: str, options_text: str, bound: dict[str, Any] | None = None, handle_of: Callable | None = None) -> Callable:
    """The function of one command, with the signature the command line's options give it (less the options in `bound`, which are set). `handle_of`: gives the handle the call works through."""
    options = parse_options(options_text)
    bound = bound or {}
    params = []
    pos = positional.rstrip("?")
    if pos and pos not in bound:
        params.append(inspect.Parameter(pos, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None if positional.endswith("?") else inspect.Parameter.empty))
    for o in options:
        if o.name in bound:
            continue
        default = inspect.Parameter.empty if o.required else (False if o.flag else None)
        params.append(inspect.Parameter(o.python, inspect.Parameter.KEYWORD_ONLY, default=default))
    params.append(inspect.Parameter("progress", inspect.Parameter.KEYWORD_ONLY, default=None))
    sig = inspect.Signature(params)
    form = _call_form(verb, noun, positional, options)
    names = {o.python: o.name for o in options}

    def command(*args, **kwargs):
        try:
            ba = sig.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{e}: the call is {form} (every option by name, as on the command line)") from None
        ba.apply_defaults()
        given = dict(bound)
        progress = ba.arguments.pop("progress")
        for k, v in ba.arguments.items():
            given[names.get(k, k)] = v
        return call(verb, noun, given, progress, None if handle_of is None else handle_of())

    words = f"{verb} {noun}".strip()
    command.__signature__ = sig
    command.__name__ = py_name(noun) if noun else py_name(verb)
    command.__qualname__ = command.__name__
    command.__module__ = f"nxr_neuro.{py_name(verb)}" if noun else "nxr_neuro"
    command.__doc__ = (f"`nxr-neuro {words}`: the command's options as arguments (`nxr-neuro {words} --help` gives each one's meaning, default and unit).\n\n"
                       f"    {form}\n\nReturns the command's result as a Result (a dict; its arrays, if any, in `.arrays`).")
    return command


# ---- the verbs as modules ----

class _Verb(types.ModuleType):
    """A verb: its nouns are its functions (`nxr_neuro.create.dataset`). A verb is not a function by itself."""

    def __call__(self, *args, **kwargs):
        nouns = ", ".join(n for n in sorted(vars(self)) if not n.startswith("_") and callable(getattr(self, n)))
        if self.__name__ == "nxr_neuro.read":
            raise TypeError("nxr_neuro.read(dataset, subject, name, ...) is retired: read a stored node by its class, nxr_neuro.read.field | statistic | operator | selection | "
                            "manifold(dataset=..., subject=..., name=..., ...) (a recording's time range as from_= and to= in seconds, or all=True)")
        raise TypeError(f"{self.__name__} is a verb, not a function: call one of its nouns ({nouns})")


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


def pipelines(*, progress=None) -> list[Pipeline]:
    """`nxr-neuro list pipelines`: the pipelines `run` takes (the shipped definitions), each with its parameters' meaning, default and unit."""
    return [Pipeline(**p) for p in json.loads(_core._pipelines())]


def verbs() -> dict[str, _Verb]:
    """One module a verb, holding one function a noun."""
    out: dict[str, _Verb] = {}
    for verb, noun, positional, options in COMMANDS:
        if not noun:
            continue
        name = py_name(verb)
        if name not in out:
            out[name] = _Verb(f"nxr_neuro.{name}", f"The verb `{verb}` of the command line: `nxr-neuro {verb} <noun>` is `nxr_neuro.{name}.<noun>(...)`.")
        f = pipelines if (verb, noun) == ("list", "pipelines") else make(verb, noun, positional, options)
        setattr(out[name], py_name(noun), f)
    out["list"].Pipeline = Pipeline
    return out


# ---- run ----

def run(pipeline: str, dataset=None, subject: str | None = None, *, from_: float | None = None, to: float | None = None, all: bool = False, replace: bool = False,
        dry_run: bool = False, preset: str | None = None, subjects: list[str] | None = None, progress: Callable[[dict], None] | None = None,
        _handle=None, **parameters) -> Result:
    """`nxr-neuro run PIPELINE --dataset D --subject S (--from S --to S | --all) [--<parameter> VALUE ...]`: a pipeline on a subject.

    The time range is REQUIRED and the caller's: `from_` and `to` in seconds on the recording's clock (both), or `all=True` (the whole recording, still read one chunk at a
    time); none is refused. The pipeline's parameters are keyword arguments (`nxr_neuro.list.pipelines()` lists each one's meaning, default and unit); `preset` names one of
    its parameter sets (the command line's `--set NAME`). `replace=True` moves another job's outputs of the same names aside (never deleted); `dry_run=True` plans only. A group
    definition runs over `subjects` (names, in order); its results go into the dataset's template (format 56: the template holds the cohort). `progress` receives each progress line as a dict. `projectors` (the command line's
    --projectors) is "active" (the default), "none", or the projectors' names (a list or comma-separated text: exactly those applied, whatever their stored status).

    Returns the result: {pipeline, subject, cache ("hit" | "computed"), stored: [{output, name, id, path}], runs, seconds}. Read what was stored with `nxr_neuro.read.<class>`.
    """
    for old, new in (("start", "from_"), ("stop", "to")):
        if old in parameters:
            raise InvalidArgumentError(f"nxr_neuro.run(..., {old}=) is retired: the time range is from_= and to= (seconds on the recording's clock), or all=True; "
                                       f"give {new}= instead of {old}=")
    if dataset is None:
        raise TypeError("nxr_neuro.run needs the dataset: run(pipeline, dataset=..., subject=..., from_=..., to=... | all=True)")
    group = subjects is not None
    if not group and subject is None:
        raise TypeError("nxr_neuro.run needs the subject: run(pipeline, dataset=..., subject=..., from_=..., to=... | all=True)")
    text = _core._run(os.fspath(dataset), subject or "", os.fspath(pipeline), json.dumps(parameters, default=_plain), None if from_ is None else float(from_),
                      None if to is None else float(to), bool(all), bool(replace), bool(dry_run), preset or "", [str(s) for s in (subjects or [])],
                      None if progress is None else (lambda line: progress(json.loads(line))), _handle)
    return Result(json.loads(text))


# ---- the handle ----

class _Bound:
    """A verb's functions with the dataset bound."""

    def __init__(self, name: str):
        self.__name__ = name

    def __repr__(self) -> str:
        return f"<{self.__name__}: {', '.join(n for n in vars(self) if not n.startswith('_'))}>"


class Dataset:
    """A dataset opened once (`nxr_neuro.open(dataset)`): its methods are the command functions with `dataset` bound (`d.show.dataset()`, `d.read.field(subject=...,
    name=..., all=True)`, `d.run("psd", subject=..., all=True)`). The library keeps the subject's database open between the handle's reading calls (read-only: it never migrates,
    recovers or locks). With mode="write" the dataset's writer lock is taken now and held until `close()` (or the end of a `with` block, or the process): every command through
    the handle borrows it, and another process's write is refused (WriterLockHeld) meanwhile. mode="read" takes no lock."""

    def __init__(self, dataset, mode: str = "read"):
        if mode not in ("read", "write"):
            raise InvalidArgumentError(f"mode is 'read' or 'write' (got {mode!r})")
        self.dataset = os.fspath(dataset)
        self.mode = mode
        self._handle = _core._Handle(self.dataset, mode == "write")
        for verb, noun, positional, options in COMMANDS:
            names = {o.name for o in parse_options(options)}
            key = "dataset" if "dataset" in names else ("folder" if positional == "folder" else None)
            if key is None or not noun:
                continue
            holder = getattr(self, py_name(verb), None)
            if holder is None:
                holder = _Bound(f"{py_name(verb)} ({self.dataset})")
                setattr(self, py_name(verb), holder)
            setattr(holder, py_name(noun), make(verb, noun, positional, options, {key: self.dataset}, self._through))
        self.compile = make("compile", "", "definition", dict((c[0], c[3]) for c in COMMANDS if not c[1])["compile"], {"dataset": self.dataset}, self._through)

    def run(self, pipeline: str, subject: str | None = None, **kwargs) -> Result:
        """`nxr_neuro.run` with the dataset bound."""
        return run(pipeline, self.dataset, subject, _handle=self._handle, **kwargs)

    def _through(self):
        return self._handle

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def close(self) -> None:
        """Close the open databases and release the writer lock (a read handle: nothing to release); the handle's calls are refused afterwards."""
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return f"<nxr_neuro.Dataset {self.dataset!r} mode={self.mode}{' closed' if self.closed else ''}>"


def open(dataset, mode: str = "read") -> Dataset:  # noqa: A001  (nxr_neuro.open, as the spec names it)
    """A handle on `dataset`: the command functions with the dataset bound; mode="write" holds the dataset's writer lock until `close()` (a context manager)."""
    return Dataset(dataset, mode)


def install(package) -> None:
    """Put the verbs on the package (`nxr_neuro.create`, ...) and in sys.modules (`from nxr_neuro.read import field`)."""
    for name, module in verbs().items():
        setattr(package, name, module)
        sys.modules[f"nxr_neuro.{name}"] = module
    package.compile = make("compile", "", "definition", dict((c[0], c[3]) for c in COMMANDS if not c[1])["compile"])
