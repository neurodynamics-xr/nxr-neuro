// nxr-compute — N-API addon entry point.
//
// Wraps the cmake-js-built addon (`nxr_compute_addon.node` at the
// project root) and exposes two surfaces in parallel:
//
//   1. Flat surface (mirrors the addon's exports verbatim): the library's functions by toolbox name, on arrays; the surface is data
//        const nxr = await initNxrCompute()
//        const surface = nxr['geometry.surfaceArrays'](verts, faces)
//        const d = nxr['geometry.heat'](surface, new Int32Array([0]))
//      (the old names - `createContext`, `solve`, `heat`, ... - are refused with an error naming the new name)
//
//   2. Six-group nested namespace `nxr.manifold.*` (preferred for new code):
//        const mctx = nxr.createManifoldContext(verts, faces)   // the surface data, made once
//        mctx.solve.eigen(300)
//        mctx.operator.connectionLaplacian({ domain: 'vertex', nSym: 4 })
//        mctx.measure.distance([0])
//        mctx.measure.distance.signed([0, 1, 2], true)
//
//      Or in functional form:
//        nxr.nxr.manifold.solve.eigen(mctx, 300)
//
// Same convention as the WASM binding's `bindings/wasm/js/index.mjs`.
// Both shells share the same conceptual surface; this shim makes the
// addon feel identical to the WASM module from a consumer's
// perspective. The underlying compute kernels are the same code — see
// the per-binding glue layers.
//
// Surface delta vs WASM (the addon doesn't yet expose every embind
// method): `uv.bff` (bff), `operator.star0` / `star2` /
// `star1Inverse`, and a few normalize/removeDC helpers throw a stable
// `Error` with code `NOT_WIRED_IN_ADDON` so consumers get a clear
// signal rather than silent undefined behaviour. To unblock these,
// add the corresponding `exports.Set(...)` line in
// `bindings/node/src/addon.cpp` and the body delegation will start
// working without further shim changes.

import path from 'node:path'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Default search path for the addon binary. cmake-js places the
// .node file under build/Release/ (or build/Debug/); scripts/build.sh
// copies it to the project root for the smoke scripts. Try the root
// first, then the build directories. Consumers can override with
// `process.env.NXR_COMPUTE_ADDON_PATH` to load a specific binary.
const repoRoot = path.resolve(__dirname, '..')
function locateAddon() {
  const explicit = process.env.NXR_COMPUTE_ADDON_PATH
  if (explicit) return explicit
  return path.join(repoRoot, 'nxr_compute_addon.node')
}

const addon = require(locateAddon())

/* ---------------------------------------------------------------------- */
/*                         Argument coercions                             */
/* ---------------------------------------------------------------------- */
//
// The addon only accepts strict TypedArray inputs (Int32Array,
// Float64Array). Plain number[] is a common JS shape, especially
// inline in tests, so coerce on the way in.

const asI32 = (arr) => arr instanceof Int32Array
  ? arr
  : Int32Array.from(arr)

const asF64 = (arr) => arr instanceof Float64Array
  ? arr
  : Float64Array.from(arr)

/* ---------------------------------------------------------------------- */
/*                      Stub helpers — uniform errors                     */
/* ---------------------------------------------------------------------- */
//
// Three classes of leaf:
//
//  • notWired(name)     — addon doesn't export this yet. Throws so
//                         downstream pattern-match on err.code works.
//  • stubWarn(name)     — placeholder leaf (matches the WASM shim's
//                         round-1 stubs). Returns a marker once and
//                         warns once per session.
//  • stubFn(name)       — same, used inside expressions.
//
// Reuses the same code names the C++ Error layer uses, so consumers
// can switch on err.code consistently across both bindings.

function notWired(name) {
  return () => {
    const err = new Error(`[NOT_WIRED_IN_ADDON] ${name} is exposed by the WASM binding but not yet by the N-API addon. Add it to bindings/node/src/addon.cpp's Init() to enable.`)
    err.code = 'NOT_WIRED_IN_ADDON'
    throw err
  }
}

const _warned = new Set()
function stubWarn(name) {
  if (!_warned.has(name)) {
    console.warn(`[nxr.manifold.${name}] is a round-1 stub — returns a TODO marker. Wire when needed.`)
    _warned.add(name)
  }
  return { method: 'todo', name }
}

/* ---------------------------------------------------------------------- */
/*                      Six-group nested context                          */
/* ---------------------------------------------------------------------- */
//
// Mirrors the structure of `bindings/wasm/js/index.mjs::makeManifoldContext`.
// The context here is a plain JS value: the surface data made once by
// `geometry.surfaceArrays` (the addon keeps nothing between calls). Each leaf
// passes it to the addon function of the same name.

function makeManifoldContext(rawCtx) {
  // Lazy JS-side memos of values (the addon keeps nothing): the bulk operator
  // assemblies, and the last eigensolve (for the spectral generators).
  let _dec  = null
  let _mesh = null
  let _eig  = null
  const dec  = () => (_dec  ??= addon['operators.assembleDECOperators'](rawCtx))
  const mesh = () => (_mesh ??= addon['operators.assembleManifoldOperators'](rawCtx))

  // ── solve ─────────────────────────────────────────────────────
  const solve = {
    poisson(sourceVerts, sourceValues) {
      const m = mesh()
      return addon['operators.poisson'](m.cotanLaplacian, m.mass, asI32(sourceVerts), asF64(sourceValues))
    },
    /** Heat diffusion via spectral evolution. Requires `solve.eigen()`
     *  to have run at least once on this context — the shim keeps the
     *  last result (the addon keeps nothing). */
    heat(sources, sourceValues, timesteps, alpha = 1.0) {
      if (!_eig) {
        const err = new Error('[NOT_PRECOMPUTED] heatDiffusion: eigenmodes not computed | hint: call solve.eigen first')
        err.code = 'NOT_PRECOMPUTED'
        throw err
      }
      const m = mesh()
      const u0 = addon['synthesis.delta'](m.nV, asI32(sources), asF64(sourceValues))
      return addon['synthesis.heatDiffusion'](m.mass, _eig, u0, asF64(timesteps), alpha)
    },
    /** Async — the addon runs the eigensolve in a libuv worker thread
     *  via Napi::AsyncWorker so the Node event loop stays free. The
     *  resolved value is normalized to `{ eigenvectors, eigenvalues,
     *  k, nConverged }` matching the WASM binding's shape — addon
     *  natively returns `{ eigenvectors, eigenvalues, nV }` which we
     *  translate. */
    async eigen(k, sigma = NaN) {
      // sigma absent or NaN: the scale-aware shift ladder (sigma = -1e-7*lmax, then -1e-4*lmax); an explicit sigma is a fixed shift.
      const m = mesh()
      const opts = { k, normalize: true, removeDC: true }
      if (Number.isFinite(sigma)) opts.sigma = sigma
      const r = await addon['operators.eigen'](m.cotanLaplacian, m.mass, opts)
      _eig = r
      return {
        eigenvectors: r.eigenvectors,
        eigenvalues:  r.eigenvalues,
        k:            r.eigenvalues.length,
        nConverged:   r.eigenvalues.length,
      }
    },
    /** Provisional placement — Hodge–Helmholtz decomposition may be
     *  reclassified to `operator.hodgeDecomp` in a future round. */
    hodge(omega) {
      return addon['signal.hodge'](rawCtx, asF64(omega))
    },
  }

  // ── operator ──────────────────────────────────────────────────
  // The split operators are CACHED — first access of any DEC piece
  // triggers a single assembleDECOperators() and reuses the result.
  // Same pattern for mass / stiffness / laplacian.
  //
  // Surface delta: addon's assembleDECOperators returns only
  // { d0, d1, hodge1 } (no hodge0, hodge2, hodge1Inverse) — those
  // four leaves are NOT_WIRED_IN_ADDON until extended in addon.cpp.
  const operator_ = {
    d0:           () => dec().d0,
    d1:           () => dec().d1,
    star0:        notWired('operator.star0'),
    star1:        () => dec().hodge1,
    star2:        notWired('operator.star2'),
    star1Inverse: notWired('operator.star1Inverse'),
    /** Per-vertex dual area (A/3-per-vertex from GC, diagonal as
     *  Float64Array — the addon's compact mass form). To get a full
     *  sparse mass matrix, add a sparse-COO export in addon.cpp. */
    mass:         () => mesh().mass,
    stiffness:    () => mesh().cotanLaplacian,
    /** Cotangent Laplacian — same matrix as `operator.stiffness()`. */
    laplacian:    () => mesh().cotanLaplacian,
    /** Connection Laplacian on the chosen domain. See
     *  bindings/wasm/js/index.d.ts ConnectionLaplacianOptions /
     *  ConnectionLaplacianResult for the option / return shapes. */
    connectionLaplacian: (options = {}) =>
      addon['operators.assembleConnectionLaplacian'](rawCtx, options),
    /** Force a fresh assembly on next access (e.g. after geometry edit). */
    invalidateCache() { _dec = null; _mesh = null },
  }

  // ── query ─────────────────────────────────────────────────────
  const query = {
    vertex(v) { return { vertexIndex: v } },
    /** The flip-out geodesic from v1 to v2 — { positions, nPoints, vertices, weights, length }; each point's
     *  vertices/weights [N×3] place it on the mesh (a vertex, an edge crossing, a face point). Parity with WASM tracePath. */
    line(v1, v2)  { return addon['geometry.tracePath'](rawCtx, v1, v2) },
    circle(v, r)  { return stubWarn('query.circle') },
    region(v, r)  { return stubWarn('query.region') },
    isoline(field, level) {
      return addon['signal.isoline'](rawCtx, asF64(field), 1, level, level)
    },
    center(sourceVerts, p = 2) {
      return addon['geometry.findCenter'](rawCtx, asI32(sourceVerts), p)
    },
  }

  // ── measure ───────────────────────────────────────────────────
  // `measure.distance` is callable AND has a `.signed` sub-property:
  //   mctx.measure.distance([0])
  //   mctx.measure.distance.signed([0,1,2], true)
  function distance(sourceVerts) {
    return addon['geometry.heat'](rawCtx, asI32(sourceVerts))
  }
  distance.signed = function distanceSigned(curveVerts, isLoop = true, levelSet = 1) {
    return addon['geometry.signedHeat'](rawCtx, asI32(curveVerts), isLoop, levelSet)
  }
  const measure = {
    distance,
    area(region)            { return stubWarn('measure.area') },
    density(region, field)  { return stubWarn('measure.density') },
    curvature()             { return addon['geometry.curvatures'](rawCtx) },
    normal(type = 0)        { return addon['geometry.normals'](rawCtx, type) },
    /** Per-face tangent frames { e1, e2, normals }, each [nF×3] row-major
     *  Float64Array. Parity with WASM `manifold.frames()`. */
    frame() { return addon['geometry.frames'](rawCtx) },
    /** Per-VERTEX tangent frames { e1, e2, normals }, each [nV×3] row-major
     *  Float64Array — the gauge of the vertex connection Laplacian (a
     *  connection coordinate a+ib at vertex i is a·e1_i + b·e2_i). Parity
     *  with WASM `manifold.vertexFrames()` and MEX 'vertexFrames'. */
    vertexFrame() { return addon['geometry.vertexFrames'](rawCtx) },
  }

  // ── uv ────────────────────────────────────────────────────────
  const uv = {
    /** bff (BFF) is exposed in the WASM binding but
     *  not in the addon yet — see surface-delta note above. */
    bff:                    notWired('uv.bff'),
    logMap(sourceVertex, strategy = 1) {
      return addon['geometry.logMap'](rawCtx, sourceVertex, strategy)
    },
  }

  // ── interpolate ───────────────────────────────────────────────
  const interpolate = {
    transport(sourceVerts, sourceVectors) {
      return addon['geometry.parallel'](rawCtx, asI32(sourceVerts), asF64(sourceVectors))
    },
    extend(sourceVerts, sourceValues) {
      return addon['geometry.extendScalar'](rawCtx, asI32(sourceVerts), asF64(sourceValues))
    },
    trivial(singVerts, singValues) {
      return addon['operators.trivial'](rawCtx, asI32(singVerts), asF64(singValues))
    },
    smoothFace(nSym = 4, alignToCurvature = false) {
      return addon['geometry.smoothFace'](rawCtx, nSym, alignToCurvature)
    },
    smoothVertex(nSym = 2, alignToCurvature = false) {
      return addon['geometry.smoothVertex'](rawCtx, nSym, alignToCurvature)
    },
  }

  return {
    // Mesh sizes, from the assembled operators (memoised on first access).
    nV: () => mesh().nV,
    nE: () => mesh().nE,
    nF: () => mesh().nF,

    /** Named-operator eigensolve (async — runs in a libuv worker, parity with
     *  WASM `manifold.eigs` but Promise-returning like addon `solve.eigen`).
     *  opts: { operator, subtype?, tau?, mass?, k, sigma?, normalize?,
     *  multiplets?, dense? } → { eigenvectors, eigenvalues, k, nConverged,
     *  blockSize }. */
    async eigs(opts) { return addon['operators.eigenOperator'](rawCtx, opts) },

    // Six-group nested namespace
    solve,
    operator: operator_,
    query,
    measure,
    uv,
    interpolate,

    /** The surface data (`geometry.surfaceArrays`). Pass it directly to any
     *  flat addon function. */
    _raw: rawCtx,
  }
}

/* ---------------------------------------------------------------------- */
/*    Functional form: `nxr.manifold.{group}.{method}(mctx, ...args)`     */
/* ---------------------------------------------------------------------- */
//
// Identical to the WASM shim's functional namespace. Each leaf
// forwards into the matching per-context method on the supplied
// ManifoldContext — both APIs share the same caches and the same
// underlying addon.

function makeFunctionalNamespace() {
  const fwd = (group, method) => (mctx, ...args) => {
    const fn = mctx?.[group]?.[method]
    if (typeof fn !== 'function') {
      const err = new Error(`nxr.manifold.${group}.${method}: not callable on this context`)
      err.code = 'INVALID_INPUT'
      throw err
    }
    return fn.apply(mctx[group], args)
  }
  return {
    solve: {
      poisson: fwd('solve', 'poisson'),
      heat:    fwd('solve', 'heat'),
      eigen:   fwd('solve', 'eigen'),
      hodge:   fwd('solve', 'hodge'),
    },
    operator: {
      d0:                  fwd('operator', 'd0'),
      d1:                  fwd('operator', 'd1'),
      star0:               fwd('operator', 'star0'),
      star1:               fwd('operator', 'star1'),
      star2:               fwd('operator', 'star2'),
      star1Inverse:        fwd('operator', 'star1Inverse'),
      mass:                fwd('operator', 'mass'),
      stiffness:           fwd('operator', 'stiffness'),
      laplacian:           fwd('operator', 'laplacian'),
      connectionLaplacian: fwd('operator', 'connectionLaplacian'),
    },
    query: {
      vertex:  fwd('query', 'vertex'),
      line:    fwd('query', 'line'),
      circle:  fwd('query', 'circle'),
      region:  fwd('query', 'region'),
      isoline: fwd('query', 'isoline'),
      center:  fwd('query', 'center'),
    },
    measure: {
      distance: Object.assign(
        (mctx, ...args) => mctx.measure.distance(...args),
        { signed: (mctx, ...args) => mctx.measure.distance.signed(...args) },
      ),
      area:      fwd('measure', 'area'),
      density:   fwd('measure', 'density'),
      curvature: fwd('measure', 'curvature'),
      normal:    fwd('measure', 'normal'),
      frame:     fwd('measure', 'frame'),
    },
    uv: {
      bff:        fwd('uv', 'bff'),
      logMap:     fwd('uv', 'logMap'),
    },
    interpolate: {
      transport:         fwd('interpolate', 'transport'),
      extend:            fwd('interpolate', 'extend'),
      trivial:    fwd('interpolate', 'trivial'),
      smoothFace:   fwd('interpolate', 'smoothFace'),
      smoothVertex: fwd('interpolate', 'smoothVertex'),
    },
  }
}

/* ---------------------------------------------------------------------- */
/*                            Public surface                              */
/* ---------------------------------------------------------------------- */

/** Synchronous initializer — returns immediately since the addon is a
 *  CommonJS native module loaded via `require()`. The async signature
 *  matches `initNxrCompute` in the WASM binding so client code can
 *  swap targets without changing call sites:
 *
 *    import { initNxrCompute } from '@neurodynamics-xr/nxr-compute/node'
 *    const nxr = await initNxrCompute()
 *
 *  Both `await nxr` and direct synchronous use are supported.
 */
export async function initNxrCompute() {
  return api
}

const api = {
  /** Flat surface: identical to the addon's raw exports, by toolbox name:
   *  `nxr['operators.eigen'](K, M, { k })` etc. */
  ...addon,

  /** Six-group nested namespace over the surface data (made once). */
  createManifoldContext(vertices, faces, options) {
    const surface = addon['geometry.surfaceArrays'](asF64(vertices), asI32(faces), options)
    return makeManifoldContext(surface)
  },

  /** Functional form of the six-group namespace. Each leaf takes a
   *  ManifoldContext as its first argument. */
  nxr: {
    manifold: makeFunctionalNamespace(),
  },
}

export default initNxrCompute
