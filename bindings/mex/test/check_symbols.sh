#!/usr/bin/env bash
# The MEX shares MATLAB's process, which ships its own HDF5, zlib and more: none of their symbols may be exported from it, and none may be a shared
# dependency of it (they are linked statically, hidden). Usage: check_symbols.sh <path to nxr_compute.mex*>. Exits 1 on a leak.
set -euo pipefail
mex="${1:?usage: check_symbols.sh <nxr_compute.mex*>}"
case "$(uname -s)" in
  Darwin) exported=$(nm -gU "$mex" | awk '{print $3}'); deps=$(otool -L "$mex" | tail -n +2 | awk '{print $1}') ;;
  *)      exported=$(nm -D --defined-only "$mex" | awk '{print $3}'); deps=$(ldd "$mex" | awk '{print $1}') ;;
esac
echo "exported symbols ($(echo "$exported" | grep -c .)):"; echo "$exported" | sed 's/^/  /'
echo "shared dependencies:"; echo "$deps" | sed 's/^/  /'
leak=$(echo "$exported" | grep -E -i '^_?(H5|Mat_|MATIO|inflate|deflate|crc32|adler32|gz[a-z]|zlib|zError|ZSTD|ZDICT|sqlite3|blosc|pugi)' || true)
if [ -n "$leak" ]; then echo "LEAK: the MEX exports symbols of its static libraries:"; echo "$leak" | head -20; exit 1; fi
# only the MEX entry points are exported
other=$(echo "$exported" | grep -v -E '^_?(mexFunction|mexfilerequiredapiversion|_fini|_init|__bss_start|_edata|_end)$' || true)
if [ -n "$other" ]; then echo "UNEXPECTED exports:"; echo "$other" | head -20; exit 1; fi
bad=$(echo "$deps" | grep -E -i 'hdf5|matio|zstd|sqlite' || true)
if [ -n "$bad" ]; then echo "UNEXPECTED shared dependencies:"; echo "$bad"; exit 1; fi
echo "OK: only the MEX entry points are exported; no HDF5, matio, zstd or SQLite shared dependency."
