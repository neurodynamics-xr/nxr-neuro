# The MATLAB face of nxr-neuro (`+nxr`)

Two layers over one MEX, `nxr_compute`:

- **The first slice (`nxr.compute`, `nxr.io`)**: the same functions, with the same names and options, as the Python wheel's first slice (`nxr_neuro.compute`,
  `nxr_neuro.io`; `docs/reference.md`). Each is one call into the C++ library on arrays; the maths is the library's.
- **The manifold package (`nxr.manifold`, `nxr.viz`)** and the `nxr_compute('command', ...)` dispatcher it wraps (about fifty commands: manifold operators,
  eigenmodes, Dirac, gauge, Delaunay). Unchanged; described below.

## Build

MATLAB R2021a or newer (name=value arguments), CMake 3.20+, Ninja, a C++17 compiler, zlib. HDF5, matio, Eigen and geometry-central are fetched or are submodules
(clone with `--recurse-submodules`), as for the main build.

    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNXR_NEURO_PORTABLE=ON \
          -DNXR_NEURO_BUILD_TESTS=OFF -DNXR_NEURO_BUILD_CLI=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    cmake --build build --target nxr_compute_mex          # build/bindings/mex/nxr_compute.mex<arch>

CMake finds MATLAB through `matlab` on the path, or `-DMatlab_ROOT_DIR=/Applications/MATLAB_R2026a.app`. (A full `cmake -S . -B build` builds the MEX too when MATLAB
is found; the options above only skip what the MEX does not need.) Add to the path:

```matlab
addpath('bindings/mex/matlab')                 % the +nxr package
addpath('build/bindings/mex')                  % nxr_compute.mex*
```

**One binary, nothing clashing.** MATLAB loads its own HDF5 and zlib. The MEX links HDF5, matio and zlib (and the C++ runtime on Linux) as static archives with hidden
symbols; it exports only `mexFunction` and `mexfilerequiredapiversion`. SQLite (io's pinned amalgamation) and zstd are in it for the pipeline level (`nxr.run`), static and hidden like the rest.
`bindings/mex/test/check_symbols.sh <mex file>` checks it with `nm` and `otool`/`ldd` (CI runs it). With a static zlib archive found (`-DZLIB_USE_STATIC_LIBS=ON`, and
`-DZLIB_ROOT=<keg>` for Homebrew's) zlib is hidden like the others; otherwise the system's shared zlib is used (macOS: two-level namespaces, so it cannot clash).

## The first API

Conventions, once: signals are `[channels x samples]`; options are name=value (`segment_samples=512`, or `'segment_samples', 512`) with the wheel's names and defaults;
**precision follows the input** (single in, single out; integers and logicals are computed as double); **indices are 1-based**, as everywhere in MATLAB (the wheel's are
0-based): faces, bins, samples, channels, the gaps' first sample, and a GIFTI triangle's vertex indices. Errors carry identifiers: `nxr:invalidArgument`, `nxr:formatError`,
`nxr:unsupported` (a valid file the reader refuses, with the reason), `nxr:io`, and the library's own (`nxr:...`). `help nxr.compute.welch.psd` gives each function's meaning,
default and unit.

| MATLAB | Python wheel | |
|---|---|---|
| `r = nxr.compute.welch.psd(x, sfreq, ...)` | `compute.welch.psd` | `r.frequencies`, `r.psd`, `r.segments`, `r.segments_skipped` |
| `r = nxr.compute.welch.cross_spectrum(x, sfreq, bins, ...)` | `compute.welch.cross_spectrum` | `bins` is `[groups x 2]`, 1-based; `r.real`, `r.imag` are `[groups x channels x channels]` |
| `y = nxr.compute.filter.lowpass(x, upper, sfreq, ...)`, `highpass(x, lower, sfreq)`, `bandpass(x, lower, upper, sfreq)`, `bandstop(...)`, `notch(x, frequencies, sfreq)` | `compute.filter.*` | |
| `r = nxr.compute.pca.pca(x, scale="none")`, `y = nxr.compute.pca.reconstruct(scores, components, mean, scale, k)` | `compute.pca.*` | `r.components`, `r.singular_values`, ... as the wheel's fields; per-component lists are rows |
| `r = nxr.compute.manifold.eigenmodes(vertices, faces, k, ...)` | `compute.manifold.eigenmodes` | faces 1-based; `r.eigenvalues` is a column |
| `rec = nxr.io.open_edf(path)`, `open_bdf`, `open_brainvision`, `open_eeglab`, `open_fif` | `io.open_*` | a `nxr.io.Recording`: `rec.header` (struct), `rec.read(first, last)`, `rec.read_all()`, `rec.close()` |
| `[x, header] = nxr.io.read_edf(path, first=1, last=1000)` (or `all=true`), `read_bdf`, `read_brainvision`, `read_eeglab`, `read_fif` | `io.read_*` | the span is always stated: `first` and `last` (inclusive), or `all=true` |
| `[data, header] = nxr.io.read_nifti(path)`, `image = nxr.io.read_gifti(path)`, `[t, names] = nxr.io.read_tsv(path)` | `io.read_nifti`, `read_gifti`, `read_tsv` | |

```matlab
rec = nxr.io.open_fif("sub-01_raw.fif");
rec.header.sampling_rate, rec.header.n_samples
x = rec.read(1, 10000);                                  % [channels x samples], SI units
y = nxr.compute.filter.bandpass(x, 8, 12, rec.header.sampling_rate);
r = nxr.compute.welch.psd(y, rec.header.sampling_rate, segment_samples=512);
plot(r.frequencies, r.psd(1, :))
```

The pipeline commands (`nxr.compute.<verb>`) are not in this slice; they wait for io's task-level API.

## The pipeline level (`nxr.run`, `nxr.list`, `nxr.read`)

`nxr.run` is `nxr-neuro run`: the same library calls (`pipelines/face.h`), so a result made here is the same cached result the command line makes and finds.

```matlab
nxr.list.pipelines()                       % the pipelines run takes; nxr.list.help('psd') prints each parameter's meaning, default and unit
nxr.list.datasets('study'); nxr.list.subjects('study/demo', where="age > 60"); nxr.list.results('study/demo', 'sub-01')
r   = nxr.run('psd', 'study/demo', 'sub-01', 'recording', 'rest', 'set', 'brainstorm', 'segment_seconds', 2, 'all', true);
r.cache, r.stored                          % 'computed' | 'hit'; what was stored (output, name, id, path)
psd = nxr.read('study/demo', 'sub-01', 'rest_psd_welch');    % psd.values [channels x frequencies], psd.meta
```

The time range is always stated: `'from'` and `'to'` (seconds) or `'all', true`; a call without one is refused (`nxr:invalidArgument`). `'progress', @(s) ...` receives each
progress line as a struct. `psd` and the shipped definitions (or a `.yaml` path) are what `run` takes; the other hand-wired pipelines are reached through the command line.

## Tests

    matlab -batch "cd bindings/mex/test/first_slice; run_first_slice_tests"     % the first slice (class-based; MEX found under build/ or in NXR_MEX_DIR)
    matlab -batch "cd bindings/mex/test; run_command_tests"                     % the nxr_compute command tests (test_*.m)

The first slice's tests use the C++ goldens (`compute/test/golden`: scipy, Brainstorm and numpy references; `io/formats/test/golden`: MNE-Python, nibabel and pandas) and
the same synthetic inputs and expected values as `bindings/python/tests`. CI (`.github/workflows/matlab.yml`) builds and tests on Linux and macOS with
`matlab-actions/setup-matlab` and `run-tests`.

---

# `+nxr/+manifold` — MATLAB six-group package

Mirrors the JS / WASM / N-API binding's `nxr.manifold.*` surface.
Each leaf is a thin wrapper around the underlying `nxr_compute('cmd', ...)`
MEX dispatcher.

## Setup

```matlab
addpath('bindings/mex/matlab')                 % this package tree
addpath('build/bindings/mex/Release')          % compiled MEX binary
```

Build the MEX with `bash scripts/build.sh Release` (requires MATLAB
R2018a+ found by CMake).

## Stateful context handle (MEX dispatcher level)

The `nxr_compute` MEX dispatcher mirrors the WASM `ContextWrapper`: a
mesh is bound **once** via `create`, and the returned `uint64` handle
keeps the geometry-central mesh, assembled operators, Cholesky factors,
and the heat / vector-heat solvers alive across calls. The integer
handle is the MATLAB analogue of WASM's `new Module.Manifold(...)`
Embind proxy.

```matlab
h = nxr_compute('create', V, F);            % build once (V: Nx3, F: Mx3, 1-based)

ops = nxr_compute('assembleManifoldOperators', h);  % cached after first call
eig = nxr_compute('solve', h, 6);                   % K,M pulled from cached ops
nxr_compute('precompute', h, 6);                    % normalized + DC-removed; caches modes

d   = nxr_compute('heat', h, 1);                    % geodesic distance (cached solver)
hd  = nxr_compute('hodge', h, omega);               % shares the CholeskyCache
cur = nxr_compute('curvatures', h);
hf  = nxr_compute('heatDiffusion', h, 1, 1.0, 0:0.1:1, 1.0);  % needs a prior solve

nxr_compute('destroy', h);                  % frees the C++ context
```

Repeated ops on one handle skip the halfedge rebuild + refactorization
that the stateless `(V, F)` convention re-pays every call. Both
conventions coexist (additive dispatch): a `uint64` scalar as the
second argument selects the handle path; `V, F` arrays select the
legacy stateless path. Handle-mode commands take **1-based** vertex /
face indices (matching the rest of the MEX marshalling).

Errors: a stale / destroyed handle raises `MException` identifier
`nxr:invalidHandle`; `heatDiffusion` / `dampedWave` before a
`solve` / `precompute` raises `nxr:notPrecomputed`.

Handle commands (handle is always the second argument):
`assembleManifoldOperators`, `assembleDECOperators`,
`assembleConnectionLaplacian`, `frames`, `normals`, `solve`,
`precompute`, `poisson`, `heat`, `tracePath`, `hodge`, `curvatures`,
`bff`, `isoline`, `trivial`, `streamline`, `whitney`, `gradient`,
`heatDiffusion`, `dampedWave`, `randomDecomposed1Form`, `parallel`,
`extendScalar`, `logMap`, `findCenter`, `signedHeat`, `smoothFace`,
`smoothVertex`, `compute`, `computeFreq`.

A MATLAB `handle`-class wrapper over this is intentionally **not**
provided here — like the JS wrappers over the WASM `Manifold`, it is an
application-side concern.

## Usage

```matlab
mctx = nxr.manifold.context(V, F);

% solve
eig = nxr.manifold.solve.eigen(mctx, 6);
oneShot = nxr.manifold.solve.precompute(mctx, 6);

% operator
K = nxr.manifold.operator.stiffness(mctx);
M = nxr.manifold.operator.mass(mctx);

% measure  (1-based indices)
d   = nxr.manifold.measure.signedDistance(mctx, [1 2 3], false);
g   = nxr.manifold.measure.distance(mctx, 1);        % geodesic from vertex 1
cur = nxr.manifold.measure.curvature(mctx);

% solve
phi = nxr.manifold.solve.poisson(mctx, [1 7], [1 -1]);
hd  = nxr.manifold.solve.hodge(mctx, omega);          % omega: nE x 1

% uv
lm = nxr.manifold.uv.logMap(mctx, 1);

% interpolate
faceField = nxr.manifold.interpolate.smoothFace(mctx, 4);
vertField = nxr.manifold.interpolate.smoothVertex(mctx, 2);
stripes   = nxr.manifold.uv.stripe(mctx, vertField.vertexFieldRaw, 3.0);

% query
center = nxr.manifold.query.center(mctx, [1 2 3]);
```

## Parity with the WASM six-group surface

Every `nxr.manifold.*` leaf the WASM binding exposes is now wired to the
MEX dispatcher via a transient context handle
(`nxr.manifold.impl.withHandle`): each leaf does `create → op → destroy`,
preserving the functional API's stateless, rebuild-per-call semantics
while reusing the dispatcher's handle commands.

Newly wired (previously `notWired`):

- `solve.poisson`, `solve.hodge`, `solve.heat` (spectral diffusion —
  runs an internal `precompute`; see its help for the `k` argument)
- `operator.d0`, `operator.d1`, `operator.star0`, `operator.star1`,
  `operator.star2`, `operator.star1Inverse`, `operator.connectionLaplacian`
- `measure.distance` (geodesic), `measure.curvature`, `measure.frame`,
  `measure.normal` (all estimators, not just the cached angle-weighted one)
- `query.isoline`
- `uv.bff`
- `interpolate.trivial` (was the misnamed `directionField.m`; the file is
  now `trivial.m`, matching the WASM `interpolate.trivial` name)

Still stubs — intentionally, because they are `stubMarker` stubs in the
WASM binding too (so this is already parity): `query.line`,
`query.circle`, `query.region`, `measure.area`, `measure.density`. They
return a `struct('method', 'todo', ...)` marker and warn once.

## API differences from JS / WASM

- **Context is a value-typed struct, not an opaque handle.** Every
  call passes `mctx` by copy. The MEX dispatcher itself is stateless;
  `mctx` carries `V, F, K, M, nV, nE, nF, totalArea, vertexDualAreas,
  vertexNormals` so every leaf has what it needs without re-assembling.
- **`measure.distance.signed` becomes a sibling.** MATLAB doesn't
  support method-attached function dispatch, so the signed-heat leaf
  is renamed to `measure.signedDistance(mctx, ...)`.
- **1-based indices (MATLAB-native).** Pass 1-based vertex / face
  indices to these wrappers, as everywhere else in MATLAB. The MEX
  marshalling converts to nxr-compute's internal 0-based convention at
  the boundary (`mxToVertexIndices` / `mxToFaceBuffer` subtract 1), so
  callers never see 0-based indices. (The JS / WASM bindings, by
  contrast, are 0-based to match three.js / Eigen / NumPy.)
- **`solve.eigen` is synchronous** here, unlike the N-API addon
  (which runs it in a libuv worker). The eigensolver releases MATLAB
  Ctrl-C via `utIsInterruptPending`.
