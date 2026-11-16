#!/usr/bin/env bash
# Build nxr-io for WebAssembly and check it against the native CLI, both ways, on a scratch datastore (synthetic octahedra only).
#   bindings/wasm-io/smoke.sh [<wasm build folder>]     needs emcc (EMSDK_PYTHON pointing at Python >= 3.10), node, and the native build in build/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W="${1:-$ROOT/build/wasm}"   # inside build/
CLI="$ROOT/build/cli/nxr-neuro"
emcmake cmake -S "$ROOT" -B "$W" -G Ninja -DNXR_NEURO_BUILD_COMPUTE=OFF -DNXR_NEURO_BUILD_BINDINGS=OFF -DNXR_NEURO_BUILD_PIPELINES=OFF -DNXR_NEURO_BUILD_CLI=OFF -DNXR_NEURO_BUILD_TESTS=OFF >/dev/null
cmake --build "$W" --target nxr_io_wasm_smoke >/dev/null
SMOKE="node $W/bindings/wasm-io/nxr_io_wasm_smoke.js"
S="$(mktemp -d "$W/smoke-store.XXXX")"
trap 'rm -rf "$S"' EXIT
# 1. WASM writes, the native CLI reads
$SMOKE write "$S" wasm | grep -q '"surfaces":\[{"name":"octahedron","values":18}\]'
"$CLI" subject list --dataset "$S/wasm" | grep -q '"name": "sub-wasm"'
# 2. the native CLI writes, WASM reads
printf 'v 0.01 0 0\nv -0.01 0 0\nv 0 0.01 0\nv 0 -0.01 0\nv 0 0 0.01\nv 0 0 -0.01\nf 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\nf 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n' > "$S/oct.obj"
"$CLI" dataset create --datastore "$S" --name native >/dev/null
"$CLI" subject create --dataset "$S/native" --name sub-native >/dev/null
"$CLI" surface add --dataset "$S/native" --subject sub-native "$S/oct.obj" >/dev/null
$SMOKE read "$S/native" | grep -q '"surfaces":\[{"name":"oct","values":18}\]'
echo "wasm-io smoke: PASS"
