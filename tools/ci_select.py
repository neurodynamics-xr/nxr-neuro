#!/usr/bin/env python3
"""CI test selection: which tests a change reaches, and which targets to build for them. Run after `cmake` configure (needs `ctest --show-only`).

    tools/ci_select.py --base SHA --head SHA [--exclude-slow]     prints CTEST_NAMES=<regex> and BUILD_TARGETS=<targets|all> (append to $GITHUB_ENV)
    tools/ci_select.py --all [--exclude-slow]                     everything (CTEST_NAMES empty, BUILD_TARGETS=all)

Rules (CONTRIBUTING.md, Continuous integration):
- a changed TEST source (`*/test/<stem>.cpp`) selects only that test; other files under a test folder (data, goldens, helpers, the .cmake) select their library's tests;
- a changed library source selects that library's tests (every kind) plus the INTEGRATION tests of the libraries that depend on it;
- a change to `tests/kinds.tsv` selects the tests whose rows it adds or edits;
- the guards (kind `guard`) always run; a shared file (top-level CMake, cmake/, this workflow, this script) selects everything.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tests as T  # noqa: E402

# library label -> libraries that depend on it (their integration tests run)
DEPENDENTS = {
    "compute": ["pipelines_graph", "pipelines", "cli"],
    "meeg": ["pipelines", "cli"],
    "io_formats": ["io", "pipelines", "cli"],
    "io": ["pipelines", "cli"],
    "pipelines_graph": ["pipelines", "cli"],
    "pipelines": ["cli"],
    "cli": [],
}
PREFIXES = [("compute/", "compute"), ("meeg/", "meeg"), ("io/formats/", "io_formats"), ("io/", "io"), ("pipelines/graph/", "pipelines_graph"),
            ("pipelines/", "pipelines"), ("cli/", "cli")]


def lib_of(f: str) -> str | None:
    return next((lab for pre, lab in PREFIXES if f.startswith(pre)), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--head"); ap.add_argument("--all", action="store_true"); ap.add_argument("--exclude-slow", action="store_true")
    a = ap.parse_args()
    ts = T.tests()
    pool = [t for t in ts if not (a.exclude_slow and "slow" in T.labels_of(t))]
    if a.all or not (a.base and a.head) or set(a.base) == {"0"}:
        print("scope: all", file=sys.stderr)
        print("CTEST_NAMES="); print("BUILD_TARGETS=all")
        return
    changed = subprocess.run(["git", "diff", "--name-only", f"{a.base}...{a.head}"], capture_output=True, text=True, check=True).stdout.split()
    names: set[str] = {t["name"] for t in pool if "guard" in T.labels_of(t)}
    allscope = False
    libs: set[str] = set()
    for f in changed:
        if f in (".github/workflows/ci.yml", "tools/ci_select.py") or f.startswith("cmake/") or ("/" not in f and not f.endswith(".md") and f not in ("LICENSE", "AUTHORS")):
            allscope = True; continue
        if f == "tests/kinds.tsv":
            # the kinds table only labels tests: the rows this change adds or edits select those tests (the guards run anyway)
            diff = subprocess.run(["git", "diff", "-U0", f"{a.base}...{a.head}", "--", f], capture_output=True, text=True, check=True).stdout
            for line in diff.splitlines():
                if line.startswith("+") and not line.startswith("+++") and line[1:].strip() and not line[1:].startswith("#"):
                    n = line[1:].split("\t")[0]
                    if any(t["name"] == n for t in pool): names.add(n)   # only a test this run may build and run (a test made slow leaves the pull-request pool)
            continue
        if f == "tools/gen_record_rows.py":
            libs.add("io"); continue
        if f.startswith(("bindings/", "docs/", "tools/", ".github/")) or f.endswith(".md"):
            continue
        lib = lib_of(f)
        if lib is None:
            allscope = True; continue
        p = Path(f)
        if "test" in p.parts and p.suffix == ".cpp":
            hit = [t["name"] for t in pool if (t.get("command") or []) and "-P" not in t["command"] and Path(t["command"][0]).name == p.stem]
            if hit:
                names.update(hit); continue
        libs.add(lib)
    if allscope:
        print("scope: all (a shared file changed)", file=sys.stderr)
        print("CTEST_NAMES="); print("BUILD_TARGETS=all")
        return
    for lib in libs:
        for t in pool:
            lab = T.labels_of(t)
            if lib in lab or ("integration" in lab and any(lib in DEPENDENTS and (lab & {d}) for d in DEPENDENTS[lib])):
                names.add(t["name"])
    sel = [t for t in pool if t["name"] in names]
    print(f"scope: {len(sel)} tests; libraries {sorted(libs)}", file=sys.stderr)
    print("CTEST_NAMES=^(" + "|".join(re.escape(n) for n in sorted(names)) + ")$")
    print("BUILD_TARGETS=" + " ".join(T.targets_for(sel)))


if __name__ == "__main__":
    main()
