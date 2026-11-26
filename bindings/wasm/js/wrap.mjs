// nxr-compute — the PUBLIC SURFACE built from a raw Embind module: `wrap(mod)`. Shared by the standalone compute build (index.mjs: createNxrComputeModule)
// and the unified nxr-neuro module (bindings/wasm-io: one module with the data client and this geometry compute), so both expose the same API.
//
// The functions are the library's, by toolbox name (`<toolbox>.<function>`; `geometry.heat` calls `nxr::geometry::heat`): one name per thing across the C++
// function, the node and the bindings. They take arrays and return arrays; nothing is kept between calls (no class, no handle, no cache). A SURFACE is data:
// the object made once by `geometry.surfaceArrays`, passed as the first argument to every surface function. An old name (`createContext`, `solve`, `heat`,
// ...) is refused with an error naming the new one (no alias).

function asI32(a) { return a instanceof Int32Array   ? a : new Int32Array(a) }
function asF64(a) { return a instanceof Float64Array ? a : new Float64Array(a) }
// a vertex array, or an Array of vertex arrays (the batch form)
const isSet = (x) => Array.isArray(x) || ArrayBuffer.isView(x)
function asI32s(a) { return Array.isArray(a) && a.length > 0 && isSet(a[0]) ? a.map(asI32) : asI32(a) }
function asF64s(a) { return Array.isArray(a) && a.length > 0 && isSet(a[0]) ? a.map(asF64) : asF64(a) }

// Old names, refused with the new one.
const RENAMED = {
  assembleManifoldOperators: 'operators.assembleManifoldOperators', assembleDECOperators: 'operators.assembleDECOperators',
  assembleConnectionLaplacian: 'operators.assembleConnectionLaplacian',
  solveEigenmodesFromTriplets: 'operators.eigen', solve: 'operators.eigen', precompute: 'operators.eigen', eigs: 'operators.eigenOperator',
  normalize: 'operators.normalize', removeDC: 'operators.removeDC', poisson: 'operators.poisson', gradient: 'operators.gradient', trivial: 'operators.trivial',
  operatorInfo: 'operators.operatorById', fieldInfo: 'operators.fieldById',
  hodge: 'signal.hodge', isoline: 'signal.isoline', streamline: 'signal.streamline', whitney: 'signal.whitney',
  curvatures: 'geometry.curvatures', bff: 'geometry.bff', frames: 'geometry.frames', vertexFrames: 'geometry.vertexFrames', normals: 'geometry.normals',
  heat: 'geometry.heat', tracePath: 'geometry.tracePath', exactDistance: 'geometry.exactDistance', parallel: 'geometry.parallel',
  extendScalar: 'geometry.extendScalar', logMap: 'geometry.logMap', findCenter: 'geometry.findCenter', signedHeat: 'geometry.signedHeat',
  smoothFace: 'geometry.smoothFace', smoothVertex: 'geometry.smoothVertex',
  heatDiffusion: 'synthesis.heatDiffusion', dampedWave: 'synthesis.dampedWave', randomDecomposed1Form: 'synthesis.randomDecomposed1Form',
}
// The context of the former surface class is gone with the class: the surface is data now.
const REMOVED = ['createContext', 'operators']

function refused(name) {
  const msg = Object.prototype.hasOwnProperty.call(RENAMED, name)
    ? `'${name}' was renamed '${RENAMED[name]}'`
    : `'${name}' was removed with the surface class: the surface is now geometry.surfaceArrays(vertices, faces) data, passed as the first argument to each function`
  return () => {
    const err = new Error(msg)
    err.code = 'RENAMED'
    throw err
  }
}

/** Builds the public surface from the raw Embind module. */
export function wrap(mod) {
  const api = {
    /** @returns {string} the nxr-compute version string, e.g. "nxr-compute 0.1.0" */
    version: () => mod.version(),

    /**
     * The surface as data, made once (the intrinsic Delaunay triangulation is the expensive step).
     *
     * @param {Float64Array | number[]} vertices `V × 3` row-major xyz triples
     * @param {Int32Array | number[]} faces `F × 3` row-major vertex indices, **0-based**
     * @param {{ intrinsicDelaunay?: boolean, mollify?: number, flipEps?: number, surface?: string }} [options]
     * @returns {Surface}
     */
    'geometry.surfaceArrays'(vertices, faces, options = null) {
      const v = asF64(vertices), f = asI32(faces)
      if (v.length % 3 !== 0) throw new Error('nxr: vertices length must be a multiple of 3')
      if (f.length % 3 !== 0) throw new Error('nxr: faces length must be a multiple of 3')
      return mod['geometry.surfaceArrays'](v, f, options)
    },

    // ── operators ────────────────────────────────────────────────
    // `variant` ∈ "mixedVoronoi" (default) | "galerkin" | "barycentric" ("lumped" is retired).
    'operators.assembleManifoldOperators': (surface, variant = '') => mod['operators.assembleManifoldOperators'](surface, variant),
    'operators.assembleDECOperators': (surface) => mod['operators.assembleDECOperators'](surface),
    'operators.assembleConnectionLaplacian': (surface, options = null) => mod['operators.assembleConnectionLaplacian'](surface, options),

    /**
     * Generalized eigensolve K φ = λ M φ for K and M as COO sparse matrices ({row, col, data, rows, cols}, as the operator assemblies return them), or any other
     * Laplacians (graph Laplacians, FEM assemblies on non-manifold meshes, custom regularized matrices).
     *
     * @param {object} K  COO matrix
     * @param {object} M  COO matrix, the same size
     * @param {object} options
     * @param {number} options.k        number of eigenmodes to compute
     * @param {number} [options.sigma]  shift-invert center. Default: none (NaN) = the scale-aware shift ladder
     *        (σ = −1e-7·λmax, then −1e-4·λmax), as the C++ does; an explicit σ is honoured as a fixed shift
     * @param {boolean} [options.normalize=false]  M-orthonormalize the eigenvectors
     * @param {boolean} [options.removeDC=false]   strip the null space
     * @param {number} [options.cancelAddr]   wasm heap pointer to cancel int32 (0 = none)
     * @param {number} [options.progressAddr] wasm heap pointer to 3×int32 progress region
     * @param {number} [options.progressLen]  progress region length, default 0
     * @returns {{ eigenvectors: Float64Array, eigenvalues: Float64Array, k: number, nConverged: number }}
     *          eigenvectors are vMajor row-major (Φ[v, k] at offset v*K + k).
     */
    'operators.eigen'(K, M, { k, sigma = NaN, normalize = false, removeDC = false, cancelAddr = 0, progressAddr = 0, progressLen = 0 } = {}) {
      if (K.rows !== K.cols || M.rows !== M.cols || K.rows !== M.rows) throw new Error('operators.eigen: K and M must be square and the same size')
      return mod['operators.eigen'](
        K.row, K.col, K.data,
        M.row, M.col, M.data,
        K.rows, k, sigma,
        cancelAddr, progressAddr, progressLen,
        normalize, removeDC,
      )
    },
    // Named-operator eigensolve. opts: { operator, subtype?, tau?, mass?, k, sigma?, normalize?, multiplets?, dense? }. Assembles the operator + its
    // natural generalized mass C++-side — no JS-side ⊗I₄. For the Dirac, pass multiplets:true for exact 4-fold multiplets, or dense:true (small meshes)
    // for an exact cross-check.
    'operators.eigenOperator': (surface, opts) => mod['operators.eigenOperator'](surface, opts),
    'operators.normalize': (U, rows, cols, M) => mod['operators.normalize'](asF64(U), rows, cols, M),
    'operators.removeDC': (eig) => mod['operators.removeDC'](eig),
    'operators.poisson': (K, M, sourceVerts, sourceValues) => mod['operators.poisson'](K, M, asI32(sourceVerts), asF64(sourceValues)),
    'operators.gradient': (surface, scalar) => mod['operators.gradient'](surface, asF64(scalar)),
    'operators.trivial': (surface, singVerts, singValues) => mod['operators.trivial'](surface, asI32(singVerts), asF64(singValues)),
    'operators.operatorById': (id) => mod['operators.operatorById'](id),
    'operators.fieldById': (id) => mod['operators.fieldById'](id),

    // ── signal ───────────────────────────────────────────────────
    // omega: a 1-form on the OPERATOR (intrinsic triangulation) mesh's edges
    'signal.hodge': (surface, omega) => mod['signal.hodge'](surface, asF64(omega)),
    'signal.isoline': (surface, scalars, numLevels = 20, minVal = 0, maxVal = 0) => mod['signal.isoline'](surface, asF64(scalars), numLevels, minVal, maxVal),
    'signal.streamline': (surface, faceField, numSeeds = 15, stepCoef = 0.15, maxSteps = 1000) =>
      mod['signal.streamline'](surface, asF64(faceField), numSeeds, stepCoef, maxSteps),
    'signal.whitney': (surface, oneForm) => mod['signal.whitney'](surface, asF64(oneForm)),

    // ── geometry ─────────────────────────────────────────────────
    'geometry.curvatures': (surface) => mod['geometry.curvatures'](surface),
    'geometry.bff': (surface) => mod['geometry.bff'](surface),
    'geometry.frames': (surface) => mod['geometry.frames'](surface),
    'geometry.vertexFrames': (surface) => mod['geometry.vertexFrames'](surface),
    'geometry.normals': (surface, type = 0) => mod['geometry.normals'](surface, type),
    // sources: a vertex array (→ one distance field) or an Array of vertex arrays (the batch form, one factorization → an Array of fields)
    'geometry.heat': (surface, sources, { tCoef = 1.0, mesh = 'intrinsic' } = {}) => mod['geometry.heat'](surface, asI32s(sources), tCoef, mesh),
    'geometry.tracePath': (surface, vStart, vEnd) => mod['geometry.tracePath'](surface, vStart, vEnd),
    'geometry.exactDistance': (surface, vStart, vEnd) => mod['geometry.exactDistance'](surface, vStart, vEnd),
    // Vector heat method (Sharp, Soliman, Crane 2019): one set, or Arrays of sets (the batch form)
    'geometry.parallel': (surface, sourceVerts, sourceVectors, tCoef = 1.0) =>
      mod['geometry.parallel'](surface, asI32s(sourceVerts), asF64s(sourceVectors), tCoef),
    'geometry.extendScalar': (surface, sourceVerts, sourceValues, tCoef = 1.0) =>
      mod['geometry.extendScalar'](surface, asI32(sourceVerts), asF64(sourceValues), tCoef),
    // a number → one result; an array of sources (the batch form, one solver) → an Array of results
    'geometry.logMap': (surface, sourceVertex, strategy = 1 /* AffineLocal */, tCoef = 1.0) =>
      mod['geometry.logMap'](surface, typeof sourceVertex === 'number' ? sourceVertex : asI32(sourceVertex), strategy, tCoef),
    'geometry.findCenter': (surface, sourceVerts, p = 2, tCoef = 1.0) => mod['geometry.findCenter'](surface, asI32(sourceVerts), p, tCoef),
    // Signed heat method (Feng & Crane 2024): a curve, or an Array of curves (the batch form)
    'geometry.signedHeat': (surface, curve, isLoop = true, levelSet = 1 /* ZeroSet */, tCoef = 1.0) =>
      mod['geometry.signedHeat'](surface, asI32s(curve), isLoop, levelSet, tCoef),
    // Smooth direction fields (Knöppel-Crane)
    'geometry.smoothFace': (surface, nSym = 4, alignToCurvature = false) => mod['geometry.smoothFace'](surface, nSym, alignToCurvature),
    'geometry.smoothVertex': (surface, nSym = 2, alignToCurvature = false) => mod['geometry.smoothVertex'](surface, nSym, alignToCurvature),

    // ── synthesis ────────────────────────────────────────────────
    'synthesis.delta': (nV, sourceVerts, sourceValues) => mod['synthesis.delta'](nV, asI32(sourceVerts), asF64(sourceValues)),
    // M: the mass (COO); eig: M-orthonormal modes (an eigensolve with normalize: true); u0: the initial condition [nV]
    'synthesis.heatDiffusion': (M, eig, u0, timesteps, alpha = 1.0) => mod['synthesis.heatDiffusion'](M, eig, asF64(u0), asF64(timesteps), alpha),
    'synthesis.dampedWave': (eig, modeIndices, amplitudes, dampings, phases, timesteps) =>
      mod['synthesis.dampedWave'](eig, asI32(modeIndices), asF64(amplitudes), asF64(dampings), asF64(phases), asF64(timesteps)),
    'synthesis.randomDecomposed1Form': (surface, alphaStrength, betaStrength, gammaStrength, seed = 42) =>
      mod['synthesis.randomDecomposed1Form'](surface, alphaStrength, betaStrength, gammaStrength, seed),

    /**
     * Create a ManifoldContext for the given mesh — the surface data (made once) under the six-group `nxr.manifold.*` namespace. The context is a plain JS
     * value; both `mctx.solve.eigen(...)` and the flat functions called with `mctx.surface` use the same surface.
     *
     * @param {Float64Array | number[]} vertices
     * @param {Int32Array | number[]} faces
     * @param {object} [options] as `geometry.surfaceArrays`
     * @returns {ManifoldContext}
     */
    createManifoldContext(vertices, faces, options = null) {
      return makeManifoldContext(api, api['geometry.surfaceArrays'](vertices, faces, options))
    },

    /**
     * Free-function form of the six-group namespace. Each leaf takes a
     * ManifoldContext as its first argument:
     *   nxr.nxr.manifold.solve.eigen(mctx, 300)
     * Backed by the same per-context methods, so behaviour is identical
     * to `mctx.solve.eigen(300)`.
     */
    nxr: {
      manifold: makeFunctionalNamespace(),
    },

    /** Underlying Embind module — escape hatch for advanced use. */
    _raw: mod,
  }
  // the old names are refused with the new one
  return new Proxy(api, {
    get(target, prop, receiver) {
      if (typeof prop === 'string' && !(prop in target) && (Object.prototype.hasOwnProperty.call(RENAMED, prop) || REMOVED.includes(prop))) return refused(prop)
      return Reflect.get(target, prop, receiver)
    },
  })
}

/* ---------------------------------------------------------------------- */
/*           Six-group nested namespace `nxr.manifold.*` shim             */
/* ---------------------------------------------------------------------- */

// Module-level dedup set so `console.warn` per stub fires once across
// the whole process, not once per ManifoldContext instance.
const _stubWarned = new Set()
function stubWarn(name) {
  if (!_stubWarned.has(name)) {
    _stubWarned.add(name)
    // eslint-disable-next-line no-console
    console.warn(
      `[nxr.manifold] ${name}: round-1 placeholder — full implementation lands in a follow-up round.`,
    )
  }
  return { method: 'todo', name }
}

/**
 * Build a ManifoldContext exposing the six-group nested namespace
 * (solve / operator / query / measure / uv / interpolate) on top of the
 * flat functions. Each leaf delegates to a single C++ function; no math is
 * reimplemented here. The split DEC and mesh operators (e.g.
 * `operator.d0`, `operator.mass`) are extracted from the memoised results
 * of the bulk `operators.assembleDECOperators` / `operators.assembleManifoldOperators`
 * calls (the C++ keeps nothing; the memo is a JS value).
 */
function makeManifoldContext(api, surface) {
  let _dec  = null
  let _mesh = null
  const dec  = () => (_dec  ??= api['operators.assembleDECOperators'](surface))
  const mesh = () => (_mesh ??= api['operators.assembleManifoldOperators'](surface, ''))

  // ----- solve --------------------------------------------------------
  const solve = {
    poisson(sourceVerts, sourceValues) {
      const m = mesh()
      return api['operators.poisson'](m.cotanLaplacian, m.mass, sourceVerts, sourceValues)
    },
    /**
     * Heat diffusion on the surface over the given timesteps: one call returns a TimeSeriesField. Stateless: it runs an internal eigensolve (k modes,
     * normalized, DC removed; default min(64, nV-1)) before the spectral diffusion — pass k to control accuracy.
     */
    heat(sources, sourceValues, timesteps, alpha = 1.0, k = undefined) {
      const m = mesh()
      const kk = k ?? Math.min(64, m.nV - 1)
      const eig = api['operators.eigen'](m.cotanLaplacian, m.mass, { k: kk, normalize: true, removeDC: true })
      const u0 = api['synthesis.delta'](m.nV, sources, sourceValues)
      return api['synthesis.heatDiffusion'](m.mass, eig, u0, timesteps, alpha)
    },
    eigen(k, sigma = NaN) {   // NaN/omitted = the shift ladder; an explicit sigma is a fixed shift
      const m = mesh()
      return api['operators.eigen'](m.cotanLaplacian, m.mass, { k, sigma })
    },
    /** Provisional — flag for future re-classification (likely
     *  `operator.hodgeDecomp` or `interpolate.hodge`). */
    hodge(omega) {
      return api['signal.hodge'](surface, omega)
    },
  }

  // ----- operator -----------------------------------------------------
  // The split operators are memoised — the first access of any DEC piece
  // triggers a single assembleDECOperators() and reuses the result for
  // every subsequent .d0() / .d1() / .star0() / .star1() / .star2() /
  // .star1Inverse() call. Same pattern for mass / stiffness / laplacian.
  const operator = {
    d0:           () => dec().d0,
    d1:           () => dec().d1,
    star0:        () => dec().hodge0,
    star1:        () => dec().hodge1,
    star2:        () => dec().hodge2,
    star1Inverse: () => dec().hodge1Inverse,
    mass:         () => mesh().mass,
    stiffness:    () => mesh().cotanLaplacian,
    /** Cotangent Laplacian — same matrix as `operator.stiffness()`. */
    laplacian:    () => mesh().cotanLaplacian,
    /**
     * Connection Laplacian on the chosen domain. Drives smoothest n-RoSy direction fields, parallel-transport energies, and tangent-bundle
     * eigendecompositions. Default `format: 'real2N'` returns a 2N×2N symmetric real sparse COO matrix that drops directly into `operators.eigen` together
     * with a block-diagonal real mass matrix `blkdiag(M, M)`; `format: 'complex'` returns a complex Hermitian COO with parallel `realData` / `imagData`.
     *
     * @param {Object}   [options]
     * @param {'vertex'|'face'|'edge'} [options.domain='vertex']
     * @param {number}   [options.nSym=1]               1=vector, 2=line, 4=cross
     * @param {number}   [options.regularization=1e-8]  additive ε·I shift
     * @param {'real2N'|'complex'} [options.format='real2N']
     */
    connectionLaplacian: (options = {}) =>
      api['operators.assembleConnectionLaplacian'](surface, options),
    /** Force a fresh assembly on next access. */
    invalidateCache() { _dec = null; _mesh = null },
  }

  // ----- query --------------------------------------------------------
  // Most query primitives are round-1 stubs. Where a clean composition
  // from existing primitives is available (e.g. `query.isoline` from
  // isoline), we wire it; otherwise emit a one-shot warning and
  // return a TODO marker.
  const query = {
    /** Identity wrapper — useful for the functional form
     *  `nxr.manifold.query.vertex(mctx, v)` where you want a structured
     *  representation of a single-vertex selection. */
    vertex(v) { return { vertexIndex: v } },
    line(v1, v2)        { return stubWarn('query.line'); /* TODO: trace flip-out geodesic between v1 and v2 */ },
    circle(v, r)        { return stubWarn('query.circle'); /* TODO: isoline of geodesic distance from v at radius r */ },
    region(v, r)        { return stubWarn('query.region'); /* TODO: vertex set within geodesic radius r of v */ },
    /** Single-level isoline of a per-vertex scalar field. */
    isoline(field, level) {
      return api['signal.isoline'](surface, field, 1, level, level)
    },
    /** Karcher mean of source vertices (vector-heat findCenter, p-th power). */
    center(sourceVerts, p = 2) {
      return api['geometry.findCenter'](surface, sourceVerts, p)
    },
  }

  // ----- measure ------------------------------------------------------
  // `measure.distance` is callable AND has a `.signed` sub-property:
  //   mctx.measure.distance([0])                  // unsigned heat-method geodesic
  //   mctx.measure.distance.signed([0,1,2], true) // signed heat distance from a curve
  function distance(sourceVerts) {
    return api['geometry.heat'](surface, sourceVerts)
  }
  distance.signed = function distanceSigned(curveVerts, isLoop = true, levelSet = 1 /* ZeroSet */) {
    return api['geometry.signedHeat'](surface, curveVerts, isLoop, levelSet)
  }
  const measure = {
    distance,
    area(region)            { return stubWarn('measure.area'); /* TODO: integrate vertex/face areas over a region selection */ },
    density(region, field)  { return stubWarn('measure.density'); /* TODO: average a scalar field over a region */ },
    curvature()             { return api['geometry.curvatures'](surface) },
    normal(type = 0)        { return api['geometry.normals'](surface, type) },
    frame()                 { return api['geometry.frames'](surface) },
  }

  // ----- uv -----------------------------------------------------------
  const uv = {
    /** Boundary First Flattening (open meshes only — throws otherwise). */
    bff() { return api['geometry.bff'](surface) },
    /** Logarithmic map at one source vertex. */
    logMap(sourceVertex, strategy = 1 /* AffineLocal */) {
      return api['geometry.logMap'](surface, sourceVertex, strategy)
    },
  }

  // ----- interpolate --------------------------------------------------
  const interpolate = {
    /** Parallel-transport tangent vectors via the vector heat method. */
    transport(sourceVerts, sourceVectors) {
      return api['geometry.parallel'](surface, sourceVerts, sourceVectors)
    },
    /** Smoothly extend a sparse scalar from the source vertices to all V. */
    extend(sourceVerts, sourceValues) {
      return api['geometry.extendScalar'](surface, sourceVerts, sourceValues)
    },
    /** Trivial-connection direction field driven by prescribed singularities. */
    trivial(singVerts, singValues) {
      return api['operators.trivial'](surface, singVerts, singValues)
    },
    /** Smoothest face-based direction field (Knöppel-Crane). */
    smoothFace(nSym = 4, alignToCurvature = false) {
      return api['geometry.smoothFace'](surface, nSym, alignToCurvature)
    },
    /** Smoothest vertex-based direction field — the `vertexFieldRaw` field
     *  in the result is the vertex direction field. */
    smoothVertex(nSym = 2, alignToCurvature = false) {
      return api['geometry.smoothVertex'](surface, nSym, alignToCurvature)
    },
  }

  return {
    // Mesh sizes (from the assembled operators, memoised)
    nV: () => mesh().nV,
    nE: () => mesh().nE,
    nF: () => mesh().nF,

    // Six-group nested namespace
    solve,
    operator,
    query,
    measure,
    uv,
    interpolate,

    /** The surface data (`geometry.surfaceArrays`): pass it to any flat function. */
    surface,

    /** Nothing to free (the surface is a JS value); kept for API compatibility. */
    dispose: () => {},
  }
}

/* ---------------------------------------------------------------------- */
/*    Functional form: `nxr.manifold.{group}.{method}(mctx, ...args)`     */
/* ---------------------------------------------------------------------- */

/**
 * Build the functional namespace tree once; every leaf forwards into
 * the corresponding per-context method on the supplied ManifoldContext.
 * Both APIs share the same memos and the same surface, so
 *   nxr.nxr.manifold.operator.d0(mctx)
 * is byte-equivalent to
 *   mctx.operator.d0()
 */
function makeFunctionalNamespace() {
  const fwd = (group, method) => (mctx, ...args) => {
    const fn = mctx?.[group]?.[method]
    if (typeof fn !== 'function') {
      throw new Error(`nxr.manifold.${group}.${method}: not callable on this context`)
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
      d0:                   fwd('operator', 'd0'),
      d1:                   fwd('operator', 'd1'),
      star0:                fwd('operator', 'star0'),
      star1:                fwd('operator', 'star1'),
      star2:                fwd('operator', 'star2'),
      star1Inverse:         fwd('operator', 'star1Inverse'),
      mass:                 fwd('operator', 'mass'),
      stiffness:            fwd('operator', 'stiffness'),
      laplacian:            fwd('operator', 'laplacian'),
      connectionLaplacian:  fwd('operator', 'connectionLaplacian'),
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
      // distance is special: callable AND has a .signed sub-leaf.
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
