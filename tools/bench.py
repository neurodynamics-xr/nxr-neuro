#!/usr/bin/env python3
"""BENCHMARKS (bench/): run the bench binaries, record the results, compare them to a baseline. Never part of the test suite, never run by CI.

    cmake -S . -B build -G Ninja -DNXR_BUILD_BENCH=ON         # once (the benchmarks are OFF by default)
    cmake --build build --target bench_wt_at bench_psd bench_isolines
    tools/bench.py run [REGEX]                  run the bench binaries (build/bench/bench_*) matching REGEX, append the results to bench-results/history.jsonl
    tools/bench.py baseline                     make the latest run the baseline (bench-results/baseline.json)
    tools/bench.py compare                      compare the latest run with the baseline: report every benchmark more than 20 % slower (median)

`run` also compares with the baseline when one exists, and exits 1 when something is more than 20 % slower. bench-results/ is in .gitignore.
Each binary prints one JSON line {name, params, runs, median_ms, min_ms} (bench/bench.h).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "bench"
OUT = ROOT / "bench-results"
HISTORY = OUT / "history.jsonl"
BASELINE = OUT / "baseline.json"
SLOWER = 0.20   # report a benchmark whose median is more than 20 % above the baseline's


def commit() -> str:
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def run_all(regex: str) -> dict:
    exes = sorted(p for p in BIN.glob("bench_*") if p.is_file() and not p.suffix and re.search(regex, p.name))
    if not exes:
        sys.exit(f"no bench binaries in {BIN} matching '{regex}': configure with -DNXR_BUILD_BENCH=ON and build the bench_* targets")
    results = {}
    for e in exes:
        r = subprocess.run([str(e)], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"{e.name} failed:\n{r.stderr[-800:]}")
        for line in r.stdout.splitlines():
            if line.startswith("{"):
                j = json.loads(line)
                results[j["name"]] = j
                print(f"{j['name']}: median {j['median_ms']:.3f} ms, min {j['min_ms']:.3f} ms ({j['params']})")
    return {"date": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "commit": commit(), "results": results}


def compare(latest: dict) -> int:
    if not BASELINE.exists():
        print("no baseline yet (tools/bench.py baseline makes the latest run the baseline)")
        return 0
    base = json.loads(BASELINE.read_text())["results"]
    slow = 0
    for name, j in latest["results"].items():
        b = base.get(name)
        if not b:
            print(f"  new (no baseline): {name}")
            continue
        ratio = j["median_ms"] / b["median_ms"] - 1
        if ratio > SLOWER:
            slow += 1
            print(f"  SLOWER {name}: {b['median_ms']:.3f} -> {j['median_ms']:.3f} ms (+{ratio * 100:.0f} %)")
    print(f"{slow} benchmark(s) more than {int(SLOWER * 100)} % slower than the baseline" if slow else "none more than 20 % slower than the baseline")
    return 1 if slow else 0


def latest_run() -> dict:
    if not HISTORY.exists():
        sys.exit("no runs yet: tools/bench.py run")
    return json.loads(HISTORY.read_text().strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("regex", nargs="?", default="")
    sub.add_parser("baseline")
    sub.add_parser("compare")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.cmd == "run":
        rec = run_all(a.regex)
        with open(HISTORY, "a") as f:
            f.write(json.dumps(rec) + "\n")
        sys.exit(compare(rec))
    if a.cmd == "baseline":
        BASELINE.write_text(json.dumps(latest_run(), indent=1))
        print("baseline set from the latest run")
        return
    sys.exit(compare(latest_run()))


if __name__ == "__main__":
    main()
