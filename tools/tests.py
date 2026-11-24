#!/usr/bin/env python3
"""THE TEST RECORD.

Every ctest run through this tool is kept in `test-results/` (in the repo, never in git: `.gitignore`):
    test-results/status.json                 the latest result of every test: status, date, commit, seconds, the lines it printed, and its FINGERPRINT
    test-results/runs/<UTC>_<commit>.json     each run, whole (and its JUnit file beside it)

A test's FINGERPRINT is what it ran: its executable (the build relinks exactly the test binaries a change reaches), the `nxr-neuro` command for a test that
calls it, the goldens and data of its library, and, for a script test (the layering guard), the tracked sources it reads. A test whose fingerprint is the
one it last passed with is still GREEN: it does not need to run again. Only STALE (passed, but what it runs has changed), FAILED and NEW tests need a run.

    tools/tests.py status REGEX         list what is green, stale, failed, new or NOT BUILT (nothing is built and nothing is run)
    tools/tests.py run REGEX            run the stale, failed and new ones among the matching tests, and record them; it NEVER builds: a test whose
                                        executable is out of date is listed as not built and left out, with the build command to ask for
    tools/tests.py build REGEX          the ONLY command that compiles: the matching tests' targets (ask the PI first: RULE 1)
    tools/tests.py status --kind unit --kind golden REGEX   only tests of these kinds (repeatable); a --kind is a selector, so REGEX may be left out
    tools/tests.py run --kind unit --fast REGEX             --fast leaves out the tests labelled `slow`
    tools/tests.py run --all REGEX      run every matching test, green or not (say why: a change to a shared core)
    tools/tests.py build --suite        build every target (end of a round only, with the PI's OK)
    tools/tests.py run --suite          run every stale, failed and new test of the full suite (end of a round only)
    tools/tests.py targets [--regex R] [--label-regex L] [--exclude-label slow]   print the build targets of a ctest selection (CI)
    tools/tests.py show TEST            the last result of one test and what it printed

NO HIDDEN BUILDS (RULE 1, the PI, 2026-11-24). Only `build` compiles; `status` and `run` never do. Whether a test's executable is out of date is
asked of the build tool without building (`ninja -n`: a dry run). A REGEX is required: it names the tests the change reaches, and `build` compiles
only their executables (and the `nxr-neuro` command, for a test that calls it). Never loop build -> run over the whole tree; never rerun a green test.

KINDS. Every test carries one kind label (tests/kinds.tsv, cmake/test_kinds.cmake): unit, golden, integration or guard (CONTRIBUTING.md, Test kinds).
`--kind K` (repeatable) keeps the tests with that label, combined with REGEX; `--fast` drops the `slow` ones. The build stays scoped to the selected tests.

Only the C++ suite (ctest). The parity pytests are not recorded here yet.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("NXR_BUILD_DIR", ROOT / "build"))
RESULTS = ROOT / "test-results"
STATUS = RESULTS / "status.json"
CLI = BUILD / "cli" / "nxr-neuro"
DATA = {"compute": ["compute/test/golden"], "meeg": ["meeg/test/golden"], "io": ["io/test/data", "io/formats/test/data", "io/formats/test/golden", "io/formats/data"], "pipelines": ["pipelines/test"]}


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def calls_cli(t: dict) -> bool:
    cmd = t.get("command") or []
    if not cmd or "-P" in cmd:
        return False
    exe = Path(cmd[0])
    try:
        lib = exe.relative_to(BUILD).parts[0]
    except ValueError:
        return False
    src = ROOT / lib / "test" / f"{exe.name}.cpp"
    return src.exists() and "std::system(" in src.read_text(errors="ignore")


def labels_of(t: dict) -> set[str]:
    for p in t.get("properties") or []:
        if p.get("name") == "LABELS":
            v = p.get("value")
            return set(v if isinstance(v, list) else [v])
    return set()


def selected(t: dict, regex: str | None, kinds: list[str] | None = None, fast: bool = False) -> bool:
    """a test matches REGEX, carries one of the kinds (when given), and is not slow (when --fast)"""
    lab = labels_of(t)
    return bool(re.search(regex or "", t["name"])) and (not kinds or bool(lab & set(kinds))) and not (fast and "slow" in lab)


KINDS = ("unit", "golden", "integration", "guard")
SEL: dict = {"kinds": None, "fast": False}   # set from the command line (--kind, --fast)


def registered_commands() -> dict[str, list[str]]:
    """name -> command of every add_test in the build's CTestTestfile.cmake files: ctest's JSON omits the command of a test whose executable is not built yet"""
    out: dict[str, list[str]] = {}
    for f in BUILD.rglob("CTestTestfile.cmake"):
        for m in re.finditer(r'^\s*add_test\(\[=\[(.+?)\]=\]\s+(.*?)\)\s*$', f.read_text(errors="ignore"), re.M):
            out[m.group(1)] = [x or y or z for x, y, z in re.findall(r'"([^"]*)"|\[=\[(.*?)\]=\]|(\S+)', m.group(2))]
    return out


def targets_for(ts: list[dict]) -> list[str]:
    """the build targets of the given ctest tests: each test's executable, and the command line for a test that calls it"""
    out: set[str] = set()
    for t in ts:
        cmd = t.get("command") or []
        if cmd and "-P" not in cmd and not re.match(r"python[\d.]*$|cmake$|sh$|bash$", Path(cmd[0]).name):   # an interpreter is never a target
            out.add(Path(cmd[0]).name)
        for a in cmd:                                  # a script test's targets are the executables it is handed (-DCLI=, -DSCENARIO=)
            if (m := re.match(r"-D(?:CLI|SCENARIO)=(.+)$", a)):
                out.add(Path(m.group(1)).name)
        if calls_cli(t):
            out.add("nxr-neuro")
    return sorted(out)


def targets_of(regex: str) -> list[str]:
    """the build targets the matching tests need"""
    return targets_for([t for t in tests() if selected(t, regex, SEL["kinds"], SEL["fast"])])


def build(regex: str | None, suite: bool) -> None:
    if os.environ.get("NXR_TESTS_NO_BUILD"):   # the caller built the targets it needs
        return
    if suite:
        cmd = ["cmake", "--build", str(BUILD)]
    else:
        tg = targets_of(regex or "")
        if not tg:
            return                              # script tests only: nothing to build
        cmd = ["cmake", "--build", str(BUILD), "--target", *tg]
    r = sh(cmd)
    if r.returncode != 0:
        sys.exit("the build failed:\n" + "\n".join(r.stdout.splitlines()[-20:] + r.stderr.splitlines()[-20:]))


def out_of_date(regex: str | None, suite: bool) -> list[str]:
    """the build targets of the selection whose build is not up to date, asked of ninja WITHOUT building (a dry run, `ninja -n`)"""
    tg = [] if suite else targets_of(regex or "")
    if not suite and not tg:
        return []
    r = sh(["ninja", "-C", str(BUILD), "-n", *tg])
    steps = [l for l in r.stdout.splitlines() if re.match(r"\[\d+/\d+\]", l)]
    if not steps:
        return []
    return tg or ["all"]


def file_hash(p: Path, h: hashlib._Hash) -> None:
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)


def tree_stamp(p: Path, h: hashlib._Hash) -> None:
    """a folder's files by name, size and modification time (goldens and data change rarely; cheap to stamp)"""
    if not p.exists():
        return
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        s = f.stat()
        h.update(f"{f.relative_to(ROOT)}:{s.st_size}:{s.st_mtime_ns}\n".encode())


_tracked = None


def tracked_sources(h: hashlib._Hash) -> None:
    """the tracked files' blob ids and the working changes: what a script test (the layering guard) reads"""
    global _tracked
    if _tracked is None:
        a = sh(["git", "-C", str(ROOT), "ls-files", "-s"]).stdout
        b = sh(["git", "-C", str(ROOT), "diff", "HEAD"]).stdout
        _tracked = (a + b).encode()
    h.update(_tracked)


def tests() -> list[dict]:
    r = sh(["ctest", "--show-only=json-v1"], cwd=BUILD)
    ts = json.loads(r.stdout)["tests"]
    if any(not t.get("command") for t in ts):             # ctest omits the command of a test whose executable is not built yet: take it from the registration
        reg = registered_commands()
        for t in ts:
            if not t.get("command") and reg.get(t["name"]):
                t["command"] = reg[t["name"]]
    return ts


def fingerprint(t: dict) -> str:
    h = hashlib.sha256()
    cmd = t.get("command") or []
    exe = Path(cmd[0]) if cmd else None
    if "-P" in cmd:                                               # a script test: the script and the sources it reads
        tracked_sources(h)
        return h.hexdigest()[:16]
    if exe and exe.exists():
        file_hash(exe, h)
        try:
            lib = exe.relative_to(BUILD).parts[0]
        except ValueError:
            lib = ""
        for d in DATA.get(lib, []):
            tree_stamp(ROOT / d, h)
        src = next((ROOT / lib / "test" / f"{exe.name}.cpp" for _ in [0] if (ROOT / lib / "test" / f"{exe.name}.cpp").exists()), None)
        if src is None:                                           # a test named apart from its source (meeg_test_x -> test_x.cpp)
            cand = list((ROOT / lib / "test").glob(f"{exe.name}.cpp")) if (ROOT / lib / "test").exists() else []
            src = cand[0] if cand else None
        if src is not None and "std::system(" in src.read_text(errors="ignore") and CLI.exists():
            file_hash(CLI, h)                                     # it runs the command line
    for a in cmd[1:]:
        p = Path(a)
        if p.is_absolute() and p.is_file():
            file_hash(p, h)
    return h.hexdigest()[:16]


def head() -> tuple[str, bool]:
    c = sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).stdout.strip()
    dirty = bool(sh(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]).stdout.strip())
    return c, dirty


def load_status() -> dict:
    return json.loads(STATUS.read_text()) if STATUS.exists() else {}


def classify(regex: str | None) -> list[tuple[str, str, str]]:
    st = load_status()
    out = []
    for t in tests():
        name = t["name"]
        if not selected(t, regex, SEL["kinds"], SEL["fast"]):
            continue
        fp = fingerprint(t)
        last = st.get(name)
        if last is None:
            state = "new"
        elif last["status"] != "passed":
            state = "failed"
        elif last["fingerprint"] != fp:
            state = "stale"
        else:
            state = "green"
        out.append((name, state, fp))
    return out


def printed(text: str) -> list[str]:
    """the lines a test prints of its own (indented measurements), not the framework's"""
    return [l.rstrip() for l in (text or "").splitlines() if l.startswith("  ") and l.strip()][:20]


def run(names: list[str], fps: dict[str, str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "runs").mkdir(exist_ok=True)
    commit, dirty = head()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    junit = RESULTS / "runs" / f"{stamp}_{commit}.xml"
    regex = "^(" + "|".join(re.escape(n) for n in names) + ")$"
    sh(["ctest", "--timeout", "300", "-R", regex, "--output-junit", str(junit)], cwd=BUILD)
    st = load_status()
    record = {"date": stamp, "commit": commit, "dirty": dirty, "tests": []}
    root = ET.parse(junit).getroot()
    for tc in root.iter("testcase"):
        name = tc.get("name") or ""
        failed = tc.find("failure") is not None or tc.find("error") is not None or tc.get("status") in ("fail", "failed")
        skipped = tc.find("skipped") is not None
        out = (tc.findtext("system-out") or "")
        r = {"status": "skipped" if skipped else ("failed" if failed else "passed"), "date": stamp, "commit": commit, "dirty": dirty,
             "seconds": float(tc.get("time") or 0), "fingerprint": fps.get(name, ""), "printed": printed(out),
             "failure": [l for l in out.splitlines() if "CHECK" in l or "FAIL" in l or "error" in l.lower()][:10] if failed else []}
        st[name] = r
        record["tests"].append({"name": name, **r})
    STATUS.write_text(json.dumps(dict(sorted(st.items())), indent=1))
    (RESULTS / "runs" / f"{stamp}_{commit}.json").write_text(json.dumps(record, indent=1))
    p = sum(1 for t in record["tests"] if t["status"] == "passed")
    f = [t["name"] for t in record["tests"] if t["status"] == "failed"]
    print(f"ran {len(record['tests'])}: {p} passed, {len(f)} failed{' (' + ', '.join(f) + ')' if f else ''}; commit {commit}{' + uncommitted changes' if dirty else ''}")
    for t in record["tests"]:
        if t["status"] == "failed":
            print(f"  FAILED {t['name']}:", *t["failure"][:5], sep="\n    ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.add_argument("regex", nargs="?"); s.add_argument("--suite", action="store_true")
    r = sub.add_parser("run"); r.add_argument("regex", nargs="?"); r.add_argument("--all", action="store_true"); r.add_argument("--suite", action="store_true")
    b = sub.add_parser("build", help="the only command that compiles: the matching tests' targets (RULE 1: ask the PI first)")
    b.add_argument("regex", nargs="?"); b.add_argument("--suite", action="store_true")
    for x in (s, r, b):
        x.add_argument("--kind", action="append", choices=KINDS, help="only tests of this kind (repeatable)")
        x.add_argument("--fast", action="store_true", help="leave out the tests labelled slow")
    w = sub.add_parser("show"); w.add_argument("test")
    g = sub.add_parser("targets", help="print the build targets of a ctest selection (for CI: build only what is tested)")
    g.add_argument("--regex", help="test names (ctest -R)")
    g.add_argument("--label-regex", help="labels (ctest -L)")
    g.add_argument("--exclude-label", action="append", default=[], help="drop tests with this label (ctest -LE; repeatable)")
    a = ap.parse_args()
    if a.cmd == "targets":
        ts = [t for t in tests() if re.search(a.regex or "", t["name"])
              and (not a.label_regex or any(re.search(a.label_regex, l) for l in labels_of(t)))
              and not (labels_of(t) & set(a.exclude_label))]
        print(" ".join(targets_for(ts)))
        return
    if a.cmd == "show":
        t = load_status().get(a.test)
        print(json.dumps(t, indent=1) if t else f"{a.test}: never run through tools/tests.py")
        return
    if not a.regex and not a.kind and not a.suite:
        sys.exit("name the tests your change reaches: tools/tests.py " + a.cmd + " REGEX or --kind KIND  (the full suite is --suite, once, at the end of a round)")
    if (a.regex or a.kind) and a.suite:
        sys.exit("--suite is the whole suite; give either a REGEX / --kind or --suite")
    SEL["kinds"], SEL["fast"] = a.kind, a.fast
    if a.cmd == "build":
        build(a.regex, a.suite)
        print("built: " + (" ".join(targets_of(a.regex or "")) if not a.suite else "every target"))
        return
    stale_build = out_of_date(a.regex, a.suite)
    if stale_build:
        ask = "tools/tests.py build " + ("--suite" if a.suite else repr(a.regex or ""))
        print(f"NOT BUILT (out of date, nothing compiled): {' '.join(stale_build)}\n  these tests' results would be of old executables; "
              f"to build them, ask the PI (RULE 1), then: {ask}")
        if a.cmd == "run":
            sys.exit(1)
    cls = classify(a.regex)
    if a.cmd == "status":
        by = {k: [n for n, s, _ in cls if s == k] for k in ("green", "stale", "failed", "new")}
        print(f"{len(cls)} tests: {len(by['green'])} green, {len(by['stale'])} stale, {len(by['failed'])} failed, {len(by['new'])} new")
        for k in ("failed", "stale", "new"):
            if by[k]:
                print(f"  {k}: " + ", ".join(by[k]))
        return
    todo = [n for n, s, _ in cls if a.all or s != "green"]
    if not todo:
        print(f"nothing to run: {len(cls)} tests green (unchanged since they passed)")
        return
    run(todo, {n: fp for n, _, fp in cls})


if __name__ == "__main__":
    main()
