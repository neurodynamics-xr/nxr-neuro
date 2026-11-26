"""The command table is the vocabulary (docs/specs/2026-11-19-io-service.md, rule 7): every command of the command line (cli/command_table.cpp) has its function,
`nxr_neuro.<verb>.<noun>`, whose arguments are the command's options (the same names with underscores, a Python keyword with a trailing underscore; the required ones
required; a flag a boolean defaulting to False; any other option defaulting to None, the command line's default), and no function is left without a command. And the
released forms that changed are refused, naming the new one."""

import inspect
import keyword
import re
from pathlib import Path

import pytest

import nxr_neuro as nn

TABLE = Path(__file__).resolve().parents[3] / "cli" / "command_table.cpp"


def py(word):
    word = word.replace("-", "_")
    return word + "_" if keyword.iskeyword(word) else word


def command_table():
    """The commands of cli/command_table.cpp, read as cli/registry.cpp reads it: {(verb, noun): (positional, optional, {option: (required, repeatable, flag)})}."""
    if not TABLE.is_file():
        pytest.skip("the repository's cli/command_table.cpp is not next to the tests")
    docs, commands, cur = {}, {}, None
    for text in re.findall(r'R"TBL\((.*?)\)TBL"', TABLE.read_text(), re.S):
        for line in text.split("\n"):
            if not line.strip() or line.startswith("#"):
                continue
            if line.startswith("--"):
                f = (line.split("\t") + [""] * 5)[:5]
                docs[f[0]] = f[1]
            elif line.startswith("@"):
                w = line[1:].split()
                cur = {"verb": w[0], "noun": w[1] if len(w) > 1 else "", "pos": "", "optional": False, "opts": []}
                commands[(cur["verb"], cur["noun"])] = cur
            elif cur is not None and ":" in line:
                key, _, val = line.partition(":")
                val = val.strip()
                if key == "pos":
                    cur["optional"] = val.endswith("?")
                    cur["pos"] = val.rstrip("?")
                elif key == "opts":
                    cur["opts"] = val.split()
    out = {}
    for (verb, noun), c in commands.items():
        path = f"{verb} {noun}".strip()
        options = {}
        for w in c["opts"]:
            required, multi = w.endswith("*"), False
            w = w.rstrip("*")
            multi = w.endswith("+")
            w = w.rstrip("+").rstrip("*").split("=")[0]
            required = required or "*" in w
            value = docs.get(f"{w}@{path}", docs.get(w))
            options[w[2:].replace("-", "_")] = (required, multi, value == "")
        out[(verb, noun)] = (c["pos"].lower(), c["optional"], options)
    return out


def function_of(verb, noun):
    if not noun:
        return getattr(nn, py(verb), None)
    return getattr(getattr(nn, py(verb), None), py(noun), None)


def test_every_command_has_its_function_with_the_same_options():
    table = command_table()
    problems = []
    commands = {k: v for k, v in table.items() if k[0] != "run"}   # `run <pipeline>`: one function, nn.run (below)
    assert len(commands) >= 75
    for (verb, noun), (positional, optional, options) in sorted(commands.items()):
        name = f"nn.{py(verb)}" + (f".{py(noun)}" if noun else "")
        f = function_of(verb, noun)
        if f is None or not callable(f):
            problems.append(f"{verb} {noun}: no function {name}")
            continue
        params = inspect.signature(f).parameters
        got_positional = [p for p in params.values() if p.kind == p.POSITIONAL_OR_KEYWORD]
        if positional:
            if [p.name for p in got_positional] != [positional]:
                problems.append(f"{name}: the positional argument is {positional}, the function has {[p.name for p in got_positional]}")
            elif (got_positional[0].default is inspect.Parameter.empty) == optional:
                problems.append(f"{name}: {positional} is {'optional' if optional else 'required'}")
        elif got_positional:
            problems.append(f"{name}: no positional argument, the function has {[p.name for p in got_positional]}")
        keywords = {p.name: p for p in params.values() if p.kind == p.KEYWORD_ONLY and p.name != "progress"}
        want = {py(o): spec for o, spec in options.items()}
        if set(keywords) != set(want):
            problems.append(f"{name}: options differ: missing {sorted(set(want) - set(keywords))}, extra {sorted(set(keywords) - set(want))}")
            continue
        for o, (required, _multi, flag) in want.items():
            default = keywords[o].default
            if required and default is not inspect.Parameter.empty:
                problems.append(f"{name}: {o} is required")
            if not required and default is inspect.Parameter.empty:
                problems.append(f"{name}: {o} is not required")
            if not required and flag and default is not False:
                problems.append(f"{name}: {o} is a flag (default False)")
            if not required and not flag and default is not None:
                problems.append(f"{name}: {o} defaults to None (the command line's default)")
    # and no function without its command
    for verb in {py(v) for v, _ in commands}:
        module = getattr(nn, verb)
        if not hasattr(module, "__dict__") or callable(module) and not inspect.ismodule(module):
            continue
        for noun, f in vars(module).items():
            if noun.startswith("_") or not callable(f) or isinstance(f, type):
                continue
            if not any(py(v) == verb and py(n) == noun for v, n in commands):
                problems.append(f"nn.{verb}.{noun}: no command `{verb} {noun}` in the command table")
    assert not problems, "\n".join(problems)


def test_run_takes_the_command_lines_range_and_flags():
    params = inspect.signature(nn.run).parameters
    assert list(params)[:3] == ["pipeline", "dataset", "subject"]
    for name in ("from_", "to", "all", "replace", "dry_run", "preset", "subjects", "progress"):
        assert params[name].kind == params[name].KEYWORD_ONLY, name
    assert ("compute", "") not in command_table()   # `compute` is reserved on the command line; nn.compute is the pure functions over arrays
    assert inspect.ismodule(nn.compute)


def test_the_old_forms_are_refused_naming_the_new(tmp_path):
    with pytest.raises(TypeError, match=r"nxr_neuro\.read\.field \| statistic \| operator \| selection \| manifold"):
        nn.read(str(tmp_path), "S1", "rest")
    with pytest.raises(nn.InvalidArgumentError, match="from_= and to="):
        nn.run("psd", str(tmp_path), "S1", start=0.0, stop=10.0)
    with pytest.raises(TypeError, match=r"nxr_neuro\.list\.results\(dataset=\.\.\., subject=\.\.\."):
        nn.list.results(str(tmp_path), "S1", "recordings")   # the kind is the noun now: nn.list.recordings
    with pytest.raises(TypeError, match=r"names"):
        nn.list.subjects(dataset=str(tmp_path), names=["S1"])   # `--subjects` is `subjects`
    with pytest.raises(TypeError, match="is a verb"):
        nn.create(dataset=str(tmp_path))
    for gone in ("Stored", "RunResult"):
        assert not hasattr(nn, gone)


def test_an_option_the_command_does_not_take_is_refused_by_the_library(tmp_path):
    # the binding's converter refuses an option it would otherwise drop (the Python signature is checked against the table above; this checks the C++ side)
    with pytest.raises(nn.InvalidArgumentError, match="unrecognized argument: bogus"):
        nn._core._command("show", "dataset", '{"dataset": "x", "bogus": 1}', None)
    with pytest.raises(nn.InvalidArgumentError, match="no command 'frobnicate dataset'"):
        nn._core._command("frobnicate", "dataset", "{}", None)
