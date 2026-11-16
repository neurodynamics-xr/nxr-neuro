#!/usr/bin/env bash
# wasmClient against the Studio's own code (parity_studio.mts) on a synthetic datastore (make_store.py).
#   NXR=<nxr checkout> bindings/wasm-io/test/parity.sh <studio checkout> [<wasm build folder>]      needs the native build (build/), node >= 22, the Studio's
#   node_modules, and nxr's environment (uv) for the synthetic Brainstorm protocols of the import test
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STUDIO="$1"
W="${2:-$ROOT/build/wasm}"
# both builds configured NOW: each stamps its results with the commit it was configured at (nxr_compute 0.1.0+<rev>, in every result's provenance and cache key),
# and the analysis comparison needs the native and the WebAssembly results stamped alike
cmake "$ROOT/build" >/dev/null 2>&1
cmake "$W" >/dev/null 2>&1
cmake --build "$ROOT/build" --target nxr-neuro >/dev/null 2>&1
cmake --build "$W" --target nxr_io_wasm >/dev/null 2>&1
T="$(mktemp -d "$W/parity.XXXX")"
trap 'rm -rf "$T"' EXIT
STORE="$(python3 "$ROOT/bindings/wasm-io/test/make_store.py" "$ROOT/build/cli/nxr-neuro" "$T")"
(cd "$STUDIO" && npx tsx "$ROOT/bindings/wasm-io/test/parity_studio.mts" "$STUDIO" "$W/bindings/wasm-io" "$STORE" \
              && npx tsx "$ROOT/bindings/wasm-io/test/parity_writes.mts" "$STUDIO" "$W/bindings/wasm-io" "$STORE")
PROTOCOLS="$(NXR="$NXR" uv run --project "$NXR" python "$ROOT/bindings/wasm-io/test/make_protocols.py" "$T/protocols" | tail -1)"
(cd "$STUDIO" && npx tsx "$ROOT/bindings/wasm-io/test/parity_imports.mts" "$STUDIO" "$W/bindings/wasm-io" "$STORE" "$PROTOCOLS" "$ROOT/build/cli/nxr-neuro")
# the ANALYSIS commands (noisecov · psd · lowpass · headmodel · inverse · sourcepsd · tiles · frames · eigen solve): native vs the module, on the session subject
NXR_IO_DETERMINISTIC=1000 "$ROOT/build/cli/nxr-neuro" convert convert "$PROTOCOLS/session" --subject S01 --condition sub-S01_ses-01_task-rest_run-01_meg --dataset "$STORE/d" >/dev/null
node "$ROOT/bindings/wasm-io/test/parity_analysis.mjs" "$W/bindings/wasm-io" "$STORE" "$ROOT/build/cli/nxr-neuro"
# the GEOMETRY COMPUTE on meshes in memory (the Manifold API, raw and wrapped), in this same module: the compute binding's smoke (expected values from the native tests)
node "$ROOT/bindings/scripts/_smoke-wasm.mjs" "$W/bindings/wasm-io/nxr_io.mjs" | tail -1
