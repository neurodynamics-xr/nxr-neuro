#!/bin/sh
# Smoke test of a built nxr-neuro: --version, --help, then a tiny synthetic dataset made by the binary itself (a dataset, a subject, an icosphere
# surface) and one computation on it (the eigenmodes of the surface). No real data. Usage: smoke.sh /path/to/nxr-neuro [work-dir]
set -eu
BIN=$1
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${2:-$(mktemp -d)}
mkdir -p "$WORK"
"$BIN" --version
"$BIN" --help > /dev/null
"$BIN" run --help > /dev/null
"$BIN" create dataset --datastore "$WORK/store" --name smoke > /dev/null
"$BIN" create subject --dataset "$WORK/store/smoke" --name s1 > /dev/null
"$BIN" add surface --dataset "$WORK/store/smoke" --subject s1 --name sphere --primary "$HERE/icosphere.obj" > /dev/null
OUT=$("$BIN" run eigenmodes --dataset "$WORK/store/smoke" --subject s1 --surface sphere --modes 6)
echo "$OUT" | tail -n 3
echo "$OUT" | grep -q '"stage": *"result"' || echo "$OUT" | grep -q result || { echo "smoke: no result line from run eigenmodes" >&2; exit 1; }
"$BIN" show subject --dataset "$WORK/store/smoke" --subject s1 > /dev/null
# a synthetic recording made by the binary (no real data), in a subject package registered in a dataset, and a time-ranged computation on it: --from and --to are required
"$BIN" create dataset --dataset "$WORK/cohort" > /dev/null
"$BIN" create package --package "$WORK/cohort/s2.nxr.zarr" --name s2 --home-dataset cohort > /dev/null
"$BIN" create recording --synthetic --package "$WORK/cohort/s2.nxr.zarr" --name rest --channels 6 --sfreq 200 --duration 30 --seed 1 > /dev/null
"$BIN" register package --dataset "$WORK/cohort" --package "$WORK/cohort/s2.nxr.zarr" --registered-by smoke > /dev/null
OUT=$("$BIN" run psd --dataset "$WORK/cohort" --subject s2 --recording rest --from 0 --to 10)
echo "$OUT" | tail -n 2
echo "$OUT" | grep -q '"stage": *"result"' || echo "$OUT" | grep -q result || { echo "smoke: no result line from run psd --from 0 --to 10" >&2; exit 1; }
echo "smoke test passed"
