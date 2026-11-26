#!/usr/bin/env bash
# Build the WebAssembly module and run the Node tests of the JavaScript face on synthetic datasets (no native build, no lab data):
#   bindings/wasm-io/test/run.sh [<wasm build folder>]        needs emcc (EMSDK_PYTHON pointing at Python >= 3.10) and node >= 22
# The coverage of the command table (test/commands.test.mjs) and the round trips (test/dataset.test.mjs) run against the built package (the build's bindings/wasm-io).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
W="${1:-$ROOT/build/wasm}"
if [ -z "${EMSDK_PYTHON:-}" ] && [ -x /opt/homebrew/bin/python3.14 ]; then export EMSDK_PYTHON=/opt/homebrew/bin/python3.14; fi   # a macOS default only where it exists (CI sets up its own)
if [ ! -f "$W/build.ninja" ]; then
  emcmake cmake -S "$ROOT" -B "$W" -G Ninja -DNXR_NEURO_BUILD_COMPUTE=ON -DNXR_NEURO_BUILD_MEEG=ON -DNXR_NEURO_BUILD_PIPELINES=ON \
    -DNXR_NEURO_BUILD_BINDINGS=OFF -DNXR_NEURO_BUILD_CLI=OFF -DNXR_NEURO_BUILD_TESTS=OFF >/dev/null
fi
cmake --build "$W" --target nxr_io_wasm >/dev/null
export NXR_WASM_PACKAGE="$W/bindings/wasm-io"
export NXR_TEST_TMP="${NXR_TEST_TMP:-${TMPDIR:-/tmp}}"
node "$ROOT/bindings/wasm-io/tools/command_table.mjs" --check
node --test "$ROOT"/bindings/wasm-io/test/*.test.mjs
