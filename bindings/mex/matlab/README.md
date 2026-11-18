# The MATLAB face of nxr-neuro (`+nxr`)

Two layers over one MEX, `nxr_compute`:

- **The first slice (`nxr.compute`, `nxr.io`)**: the same functions, with the same names and options, as the Python wheel's first slice (`nxr_neuro.compute`,
  `nxr_neuro.io`; `docs/reference.md`). Each is one call into the C++ library on arrays; the maths is the library's.
- **The manifold package (`nxr.manifold`, `nxr.viz`)** and the `nxr_compute('<toolbox>.<function>', ...)` dispatcher it wraps (the library's functions by their
  toolbox names, on arrays and the surface as data). Described below.

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
`nxr:unsupported` (a valid file the reader refuses, with the reason), `nxr:io`, and the library's own (`nxr:...`). `help nxr.compute.signal.psd` gives each function's meaning,
default and unit.

| MATLAB | Python wheel | |
|---|---|---|
| `r = nxr.compute.signal.psd(x, sfreq, ...)` | `compute.signal.psd` | `r.frequencies`, `r.psd`, `r.segments`, `r.segments_skipped` |
| `r = nxr.compute.signal.xpsd(x, sfreq, bins, ...)` | `compute.signal.xpsd` | `bins` is `[groups x 2]`, 1-based; `r.real`, `r.imag` are `[groups x channels x channels]` |
| `y = nxr.compute.signal.lowpass(x, upper, sfreq, ...)`, `highpass(x, lower, sfreq)`, `bandpass(x, lower, upper, sfreq)`, `bandstop(...)`, `notch(x, frequencies, sfreq)` | `compute.signal.*` | |
| `r = nxr.compute.stats.pca(x, scale="none")`, `y = nxr.compute.stats.reconstruct(scores, components, mean, scale, k)` | `compute.pca.*` | `r.components`, `r.singular_values`, ... as the wheel's fields; per-component lists are rows |
| `r = nxr.compute.operators.eigen(vertices, faces, k, ...)` | `compute.operators.eigen` | faces 1-based; `r.eigenvalues` is a column |
| `rec = nxr.io.open_edf(path)`, `open_bdf`, `open_brainvision`, `open_eeglab`, `open_fif` | `io.open_*` | a `nxr.io.Recording`: `rec.header` (struct), `rec.read(first, last)`, `rec.read_all()`, `rec.close()` |
| `[x, header] = nxr.io.read_edf(path, first=1, last=1000)` (or `all=true`), `read_bdf`, `read_brainvision`, `read_eeglab`, `read_fif` | `io.read_*` | the span is always stated: `first` and `last` (inclusive), or `all=true` |
| `[data, header] = nxr.io.read_nifti(path)`, `image = nxr.io.read_gifti(path)`, `[t, names] = nxr.io.read_tsv(path)` | `io.read_nifti`, `read_gifti`, `read_tsv` | |

```matlab
rec = nxr.io.open_fif("sub-01_raw.fif");
rec.header.sampling_rate, rec.header.n_samples
x = rec.read(1, 10000);                                  % [channels x samples], SI units
y = nxr.compute.signal.bandpass(x, 8, 12, rec.header.sampling_rate);
r = nxr.compute.signal.psd(y, rec.header.sampling_rate, segment_samples=512);
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

# `+nxr/+manifold`: the surface functions

Each leaf is a thin wrapper around the underlying `nxr_compute('<toolbox>.<function>', ...)` MEX dispatcher.

## Setup

```matlab
addpath('bindings/mex/matlab')                 % this package tree
addpath('build/bindings/mex/Release')          % compiled MEX binary
```

Build the MEX with `bash scripts/build.sh Release` (requires MATLAB R2018a+ found by CMake).

## The MEX dispatcher: the library's functions by toolbox name, on arrays

Every command is the library's function, named by its toolbox and its own word (`geometry.heat` calls `nxr::geometry::heat`; one name per thing across the C++
function, the node and the bindings). Every command takes arrays and returns arrays, and **nothing is kept between calls**: there is no context, no handle, no cache.

**The surface is data.** `geometry.surfaceArrays(V, F)` makes it once (the intrinsic Delaunay triangulation is the expensive step) and returns a struct: `positions`,
`faces` (1-based) and the `triangulation` (its options, its record and geometry-central's own arrays, kept raw and opaque). Every surface command takes that struct as
its first argument and rebuilds the library's `SurfaceArrays` from it. A factorization a command needs is made inside the command.

```matlab
S   = nxr_compute('geometry.surfaceArrays', V, F);          % once; opts: intrinsicDelaunay, mollify, flipEps, surface
ops = nxr_compute('operators.assembleManifoldOperators', S);
eig = nxr_compute('operators.eigen', ops.cotanLaplacian, ops.mass, 6, [], true, true);   % K, M, k, sigma ([] = shift ladder), normalize, removeDC
d   = nxr_compute('geometry.heat', S, 1);                   % geodesic distance; a cell array of source sets gives one column per set
hd  = nxr_compute('signal.hodge', S, omega);                % omega: a 1-form on the input mesh's edges
cur = nxr_compute('geometry.curvatures', S);
u0  = nxr_compute('synthesis.delta', ops.nV, 1, 1.0);
hf  = nxr_compute('synthesis.heatDiffusion', ops.mass, eig, u0, 0:0.1:1, 1.0);
```

Commands (indices 1-based):

- `geometry.`: `surfaceArrays`, `fixDelaunay`, `parallel`, `extendScalar`, `logMap`, `findCenter`, `signedHeat`, `heat`, `smoothFace`, `smoothVertex`, `compute`
  and `computeFreq` (stripe patterns), `frames`, `vertexFrames`, `normals`, `curvatures`, `bff`, `tracePath`.
- `operators.`: `assembleManifoldOperators`, `assembleDECOperators`, `assembleConnectionLaplacian`, `assembleTrivialConnectionLaplacian`, `eigen`, `normalize`, `removeDC`,
  `poisson` (K, M, vertices, values), `trivial`, `gradient`, `frameTransport`, `liftToWorld`, `liftToFrame`, `operatorById`, `fieldById`.
- `signal.`: `hodge`, `isoline`, `streamline`, `whitney`.
- `synthesis.`: `delta`, `heatDiffusion`, `dampedWave`, `randomDecomposed1Form`.
- Batch forms where the library has them: `geometry.heat` and `geometry.signedHeat` take a cell array of source sets (or curves), `geometry.parallel` cell arrays of
  sources and vectors, `geometry.logMap` a vector of sources (a struct array back).
- A 1-form (`signal.hodge`, `signal.whitney`, `synthesis.randomDecomposed1Form`) is indexed by the input mesh's edges.

**Old names are refused, never aliased.** `nxr_compute('heat', ...)` raises `nxr:renamed`: "The command "heat" was renamed: use "geometry.heat"" (likewise `solve`
and `precompute`, now `operators.eigen` with its `normalize` and `removeDC` arguments; `hodge`, now `signal.hodge`; and so on). The former context commands
(`create`, `destroy`, `topology`, `geometry`, `gauge`, `bundle`, `operators`, `embedded`, `intrinsic`, `extrinsic`, `facets`) went with the surface class and
are refused with the reason: the surface is now the data of `geometry.surfaceArrays`, passed to each command. `version` is unchanged.

Errors: library errors carry their identifier (`nxr:invalidInput`, `nxr:cancelled`, ...); `operators.eigen` honours Ctrl-C.

## Usage of the wrappers

```matlab
mctx = nxr.manifold.context(V, F);          % the surface data, K and M, made once

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

Still stubs (as in the WASM binding): `query.line`, `query.circle`, `query.region`, `measure.area`, `measure.density`. They return a `struct('method', 'todo', ...)`
marker and warn once.

## API differences from JS / WASM

- **The context is a value-typed struct, not an opaque handle.** Every call passes `mctx` by copy; `mctx` carries `surface` (the `geometry.surfaceArrays` struct),
  `V, F, K, M, nV, nE, nF, totalArea, vertexDualAreas, vertexNormals`, so every leaf has what it needs without re-assembling. The MEX dispatcher itself keeps nothing.
- **`measure.distance.signed` becomes a sibling.** MATLAB doesn't support method-attached function dispatch, so the signed-heat leaf is renamed to
  `measure.signedDistance(mctx, ...)`.
- **1-based indices (MATLAB-native).** Pass 1-based vertex / face indices to these wrappers, as everywhere else in MATLAB. The MEX marshalling converts to
  nxr-compute's internal 0-based convention at the boundary (`mxToVertexIndices` / `mxToFaceBuffer` subtract 1), so callers never see 0-based indices. (The JS / WASM
  bindings, by contrast, are 0-based to match three.js / Eigen / NumPy.)
- **`solve.eigen` is synchronous** here, unlike the N-API addon (which runs it in a libuv worker). The eigensolver releases MATLAB Ctrl-C via `utIsInterruptPending`.
