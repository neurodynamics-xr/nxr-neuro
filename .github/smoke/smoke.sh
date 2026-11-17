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
"$BIN" compute --help > /dev/null
"$BIN" create dataset --datastore "$WORK/store" --name smoke > /dev/null
"$BIN" create subject --dataset "$WORK/store/smoke" --name s1 > /dev/null
"$BIN" add surface --dataset "$WORK/store/smoke" --subject s1 --name sphere --primary "$HERE/icosphere.obj" > /dev/null
OUT=$("$BIN" compute eigenmodes --dataset "$WORK/store/smoke" --subject s1 --surface sphere --modes 6)
echo "$OUT" | tail -n 3
echo "$OUT" | grep -q '"stage": *"result"' || echo "$OUT" | grep -q result || { echo "smoke: no result line from compute eigenmodes" >&2; exit 1; }
"$BIN" show subject --dataset "$WORK/store/smoke" --subject s1 > /dev/null
echo "smoke test passed"
