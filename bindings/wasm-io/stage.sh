#!/usr/bin/env bash
# Build nxr-neuro's WebAssembly module and stage it as a LOCAL npm package (no release, no registry) for development:
#   bindings/wasm-io/stage.sh [<wasm build folder>]          → <build>/package, printed at the end
# The Studio (or any Node project) then uses it with `npm link <that folder>` (no change to its package.json), or a `file:` dependency while trying it.
# Re-run after a change: the linked folder is refreshed in place.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W="${1:-$ROOT/build/wasm}"   # inside build/
export EMSDK_PYTHON="${EMSDK_PYTHON:-/opt/homebrew/bin/python3.14}"
if [ ! -f "$W/build.ninja" ]; then
  emcmake cmake -S "$ROOT" -B "$W" -G Ninja -DNXR_NEURO_BUILD_COMPUTE=ON -DNXR_NEURO_BUILD_MEEG=ON -DNXR_NEURO_BUILD_PIPELINES=ON \
    -DNXR_NEURO_BUILD_BINDINGS=OFF -DNXR_NEURO_BUILD_CLI=OFF -DNXR_NEURO_BUILD_TESTS=OFF >/dev/null
fi
cmake --build "$W" --target nxr_io_wasm >/dev/null
P="$W/package"
mkdir -p "$P"
for f in index.mjs index.d.ts commands.mjs commands.d.ts arrays.mjs compute.mjs nxr_io.mjs nxr_io.wasm; do cp "$W/bindings/wasm-io/$f" "$P/$f"; done
VERSION="$(sed -n 's/^project(nxr_neuro VERSION \([0-9.]*\).*/\1/p' "$ROOT/CMakeLists.txt" | head -1)"
REV="$(git -C "$ROOT" rev-parse --short=10 HEAD)$(git -C "$ROOT" diff --quiet || echo -dirty)"
# the version names the commit it was built from (a local build: 0.1.0-dev.<rev>)
sed -e "s/\"version\": \"[^\"]*\"/\"version\": \"${VERSION:-0.1.0}-dev.${REV}\"/" "$ROOT/bindings/wasm-io/package.json" > "$P/package.json"
echo "$P"
