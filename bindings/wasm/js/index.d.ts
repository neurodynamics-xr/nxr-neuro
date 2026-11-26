// TypeScript declarations for the nxr-compute WASM binding.
// See docs/nxr-compute/three-js-integration.md and docs/nxr-compute/visualization-recipes.md
// for usage patterns.

/** Sparse matrix in COO triplet form. Always real double. */
export interface SparseMatrixCOO {
  row:  Int32Array
  col:  Int32Array
  data: Float64Array
  rows: number
  cols: number
  nnz:  number
}

/** Sparse complex matrix in COO triplet form. Used by the
 *  connection-Laplacian "complex" output format — `realData` and
 *  `imagData` carry the real and imaginary parts of each entry. */
export interface SparseMatrixCOOComplex {
  row:       Int32Array
  col:       Int32Array
  realData:  Float64Array
  imagData:  Float64Array
  rows:      number
  cols:      number
  nnz:       number
}

/** Connection-Laplacian assembly options. All fields are optional;
 *  defaults are { domain: 'vertex', nSym: 1, regularization: 1e-8,
 *  format: 'real2N' }. */
export interface ConnectionLaplacianOptions {
  domain?:         'vertex' | 'face' | 'edge'
  /** Symmetry order. 1 = vector field, 2 = line field, 4 = cross field. Must be > 0. */
  nSym?:           number
  /** Additive ε·I shift. Default 1e-8. */
  regularization?: number
  format?:         'real2N' | 'complex'
}

/** Connection Laplacian on a chosen domain.
 *  - For `format: 'real2N'`, `K` is a 2N×2N symmetric real COO matrix
 *    (each complex entry a + bi lowered to the 2×2 block [[a,-b],[b,a]]).
 *    Drops directly into `operators.eigen` with a
 *    block-diagonal real mass matrix `blkdiag(M, M)`.
 *  - For `format: 'complex'`, `K` is an N×N complex Hermitian COO
 *    matrix with parallel `realData` / `imagData` value arrays. */
export type ConnectionLaplacianResult =
  | {
      K:              SparseMatrixCOO
      baseDim:        number
      outputDim:      number   // 2 * baseDim
      domain:         'vertex' | 'face' | 'edge'
      nSym:           number
      regularization: number
      format:         'real2N'
    }
  | {
      K:              SparseMatrixCOOComplex
      baseDim:        number
      outputDim:      number   // baseDim
      domain:         'vertex' | 'face' | 'edge'
      nSym:           number
      regularization: number
      format:         'complex'
    }

/** FEM mass-matrix variant string passed to assembleManifoldOperators(): "mixedVoronoi" (the default: Meyer et al. 2003, of the
 *  mollified input, recorded before the flips), "galerkin", or "barycentric" (A/3 per vertex; formerly "lumped", now retired). */
export type MassMatrixVariant = "mixedVoronoi" | "galerkin" | "barycentric"

export interface ManifoldOperators {
  /** Cotangent Laplacian (PSD, symmetrized), V×V sparse */
  stiffness:   SparseMatrixCOO
  /** Mass matrix, V×V sparse. SPD; diagonal for "mixedVoronoi" and "barycentric",
   *  has off-diagonal couplings for "galerkin". */
  mass:        SparseMatrixCOO
  /** Which variant produced `mass`. */
  massVariant: MassMatrixVariant
  /** Per-vertex dual area (A/3-per-vertex from GC), V — always sourced
   *  from `vertexDualAreas` regardless of `massVariant`. */
  vertexAreas: Float64Array
  /** Per-vertex normals, V*3 row-major */
  normals:     Float64Array
  totalArea:   number
  nV:          number
  nE:          number
  nF:          number
}

export interface DECOperators {
  /** Vertex → edge gradient, E×V sparse */
  d0:            SparseMatrixCOO
  /** Edge → face curl,     F×E sparse */
  d1:            SparseMatrixCOO
  hodge0:        SparseMatrixCOO
  hodge1:        SparseMatrixCOO
  hodge2:        SparseMatrixCOO
  hodge1Inverse: SparseMatrixCOO
}

export interface EigenResult {
  /** Eigenvectors stored row-major as V*K floats. Column k starts at offset k*nV in this flat buffer. */
  eigenvectors: Float64Array
  /** Eigenvalues, length K, sorted ascending. */
  eigenvalues:  Float64Array
  k:            number
  nConverged:   number
}

/** Options for `operators.eigenOperator` — the named-operator eigensolve. */
export interface EigsOptions {
  /** Which operator to eigensolve. 'laplacian' pairs with `subtype`; 'cotan'/
   *  'graph' are convenience aliases; 'dirac'/'diracFace' are the [4·domain]
   *  quaternionic families (blockSize 4). */
  operator: "laplacian" | "cotan" | "graph" | "dirac" | "diracFace"
  /** Required when operator === 'laplacian'. */
  subtype?: "cotan" | "graph"
  /** Blend for dirac/diracFace ∈ [0,1] (0 = intrinsic, 1 = pure extrinsic). */
  tau?:     number
  /** Vertex-domain mass for cotan/dirac (default 'galerkin'). */
  mass?:    MassMatrixVariant
  /** Number of eigenpairs. */
  k:        number
  /** Shift for shift-invert. Omitted = the scale-aware shift ladder (σ = −1e-7·λmax, then −1e-4·λmax); an explicit σ is a fixed shift. */
  sigma?:   number
  /** M-orthonormalize the basis (default true). */
  normalize?:  boolean
  /** Dirac only: reconstruct exact 4-fold multiplets from the right-ℍ symmetry.
   *  The returned count is rounded up to a multiple of 4. */
  multiplets?: boolean
  /** Exact dense generalized solve — small meshes / verification only. */
  dense?:      boolean
}

/** Result of `operators.eigenOperator` — EigenResult plus the operator's block size
 *  (1 scalar / 4 Dirac), so consumers can reshape the n·k eigenvector buffer
 *  (n = blockSize · domainSize). */
export interface EigsResult extends EigenResult {
  blockSize: number
}

export interface FaceFrames {
  /** First per-face tangent vector, F*3 row-major */
  e1:      Float64Array
  /** Second per-face tangent vector (n × e1), F*3 row-major */
  e2:      Float64Array
  /** Per-face normal, F*3 row-major */
  normals: Float64Array
}

export interface PrecomputeResult {
  operators:  ManifoldOperators
  dec:        DECOperators
  eigenmodes: EigenResult
  frames: FaceFrames
}

export interface HodgeResult {
  /** Per-vertex exact (gradient) potential α */
  exactPotential:    Float64Array
  /** Per-vertex co-exact (curl) potential β (averaged from face) */
  coExactPotentialV: Float64Array
  combinedPotential: Float64Array
  /** Per-edge dα 1-form */
  dAlpha:            Float64Array
  /** Per-edge δβ 1-form */
  deltaBeta:         Float64Array
  /** Per-edge harmonic γ 1-form */
  gamma:             Float64Array
  /** Face-centered 3D vector fields, each F*3 row-major */
  omegaVectors:      Float64Array
  dAlphaVectors:     Float64Array
  deltaBetaVectors:  Float64Array
  gammaVectors:      Float64Array
}

export interface CurvatureResult {
  gaussian:     Float64Array
  mean:         Float64Array
  kMin:         Float64Array
  kMax:         Float64Array
  /** Per-vertex max-curvature direction, V*3 row-major */
  principalDir: Float64Array
}

export interface PolylineResult {
  /** N*3 row-major points */
  positions: Float64Array
  nPoints:   number
}

export interface SegmentsResult {
  /** (2*segmentCount)*3 row-major endpoint pairs */
  positions:    Float64Array
  segmentCount: number
}

export interface DirectionFieldResult {
  /** Per-edge connection 1-form */
  connections:          Float64Array
  /** F*3 per-face direction vectors */
  directionVectors:     Float64Array
  /** F*3 per-face 90°-rotated direction vectors */
  orthogonalVectors:    Float64Array
  eulerCharacteristic:  number
  gaussBonnetSatisfied: boolean
}

export interface TimeSeriesField {
  /** T*nV row-major frames; frame ti starts at offset ti*nV */
  data: Float32Array
  T:    number
  nV:   number
}

export interface LogMapResult {
  /** V*2 row-major (logX, logY) coordinates in source's tangent frame.
   *  norm is geodesic distance from source; atan2 is bearing. */
  logCoords: Float64Array
  /** Source vertex's tangent basis e1 in world coords (3 floats). */
  sourceE1:  Float64Array
  /** Source vertex's tangent basis e2 in world coords (3 floats). */
  sourceE2:  Float64Array
}

export interface SmoothVertexFieldResult {
  /** V*3 row-major lifted world-space vectors (principal nSym-RoSy representative). */
  vertexVectors:  Float64Array
  /** V*2 row-major raw nSym-RoSy field in vertex tangent basis.
   *  Pass back into compute as-is. */
  vertexFieldRaw: Float64Array
  nSym: number
}

/** Strategy for VHM log-map. AffineLocal recommended (fast, accurate near source). */
export enum LogMapStrategy {
  VectorHeat     = 0,
  AffineLocal    = 1,
  AffineAdaptive = 2,
}

export enum SignedHeatLevelSet {
  None     = 0,
  ZeroSet  = 1,
  Multiple = 2,
}

/** Vertex-normal estimator selection. */
export enum NormalType {
  AngleWeighted   = 0,
  AreaWeighted    = 1,
  EqualWeighted   = 2,
  SphereInscribed = 3,
  MeanCurvature   = 4,
  GaussCurvature  = 5,
}

/**
 * A SURFACE is data: the object made once by `geometry.surfaceArrays` (positions, faces and the intrinsic Delaunay triangulation, the expensive step).
 * Every surface function takes it as its first argument and rebuilds the library's SurfaceArrays from it; nothing is kept between calls. The triangulation's
 * arrays are opaque: pass the object on, do not edit it.
 */
export interface Surface {
  /** [3 nV] row-major */
  positions: Float64Array
  /** [3 nF], 0-based */
  faces: Int32Array
  triangulation: {
    options: { intrinsic: boolean; mollify: number; flipEps: number; mass: string; surface: string }
    record: {
      mesh: string; mollify: number; epsilon: number; flipEps: number; flips: number; delaunay: boolean; mass: string
      boundaryVertices: number; geometryCentral: string; fingerprint: string; gauge: string; gaugeId: string
    }
    mass: Float64Array
    inputEdgeLength: Float64Array
    edgeLength: Float64Array
    signpost: Float64Array
    vertexAngleSum: Float64Array
    halfedgeVectorInVertex: Float64Array
    /** geometry-central's own 0-based ids (-1 = none), as exact integers in a Float64Array */
    halfedgeNext: Float64Array
    halfedgeVertex: Float64Array
    halfedgeFace: Float64Array
    vertexHalfedge: Float64Array
    faceHalfedge: Float64Array
    boundaryLoops: number
  }
}

export interface SurfaceOptions {
  /** false = the named option "embedded" (the input as it is); default true (the intrinsic mollified Delaunay mesh) */
  intrinsicDelaunay?: boolean
  /** relative mollification margin (default 1e-6) */
  mollify?: number
  /** an edge is flipped while its cotangent weight is below -flipEps (default 1e-6) */
  flipEps?: number
  /** an id, recorded in the gauge identity */
  surface?: string
}

type Ints = Int32Array | number[]
type Floats = Float64Array | number[]

export interface EigenOptions {
  k: number
  /** omitted/NaN = the shift ladder; an explicit sigma is a fixed shift */
  sigma?: number
  /** M-orthonormalize the eigenvectors (default false) */
  normalize?: boolean
  /** strip the null space (default false) */
  removeDC?: boolean
  cancelAddr?: number
  progressAddr?: number
  progressLen?: number
}

/**
 * The module: the library's functions by their toolbox name, `<toolbox>.<function>` (`geometry.heat` calls `nxr::geometry::heat`). They take arrays and return
 * arrays; nothing is kept between calls. An old name (`createContext`, `solve`, `heat`, ...) is refused with an error naming the new one.
 */
export interface NxrCompute {
  version(): string

  'geometry.surfaceArrays'(vertices: Float64Array | number[], faces: Int32Array | number[], options?: SurfaceOptions): Surface

  'operators.assembleManifoldOperators'(surface: Surface, variant?: MassMatrixVariant | ""): ManifoldOperators
  'operators.assembleDECOperators'(surface: Surface): DECOperators
  'operators.assembleConnectionLaplacian'(surface: Surface, options?: ConnectionLaplacianOptions): ConnectionLaplacianResult
  /** K phi = lambda M phi for K, M as COO sparse matrices (from `assembleManifoldOperators` or any other Laplacian). */
  'operators.eigen'(K: SparseMatrixCOO, M: SparseMatrixCOO, options: EigenOptions): EigenResult
  /** a named operator's eigenproblem, assembled C++-side on the surface */
  'operators.eigenOperator'(surface: Surface, opts: EigsOptions): EigsResult
  'operators.normalize'(U: Float64Array, rows: number, cols: number, M: SparseMatrixCOO): Float64Array
  'operators.removeDC'(eig: EigenResult): EigenResult
  'operators.poisson'(K: SparseMatrixCOO, M: SparseMatrixCOO, sourceVerts: Ints, sourceValues: Floats): Float64Array
  'operators.gradient'(surface: Surface, scalar: Float64Array): Float64Array
  'operators.trivial'(surface: Surface, singVerts: Ints, singValues: Floats): DirectionFieldResult
  'operators.operatorById'(id: string): Record<string, unknown>
  'operators.fieldById'(id: string): Record<string, unknown>

  /** omega: a 1-form on the operator (intrinsic triangulation) mesh's edges */
  'signal.hodge'(surface: Surface, omega: Float64Array): HodgeResult
  'signal.isoline'(surface: Surface, scalars: Float64Array, numLevels?: number, minVal?: number, maxVal?: number): SegmentsResult
  'signal.streamline'(surface: Surface, faceField: Float64Array, numSeeds?: number, stepCoef?: number, maxSteps?: number): SegmentsResult
  'signal.whitney'(surface: Surface, oneForm: Float64Array): Float64Array

  'geometry.curvatures'(surface: Surface): CurvatureResult
  /** Boundary First Flattening: throws on closed meshes */
  'geometry.bff'(surface: Surface): Float64Array
  'geometry.frames'(surface: Surface): FaceFrames
  'geometry.vertexFrames'(surface: Surface): FaceFrames
  'geometry.normals'(surface: Surface, type?: NormalType): Float64Array
  /** a vertex array gives one distance field; an Array of vertex arrays (the batch form, one factorization) gives an Array of them */
  'geometry.heat'(surface: Surface, sources: Ints, options?: { tCoef?: number; mesh?: 'intrinsic' | 'input' }): Float64Array
  'geometry.heat'(surface: Surface, sources: Ints[], options?: { tCoef?: number; mesh?: 'intrinsic' | 'input' }): Float64Array[]
  'geometry.tracePath'(surface: Surface, vStart: number, vEnd: number): PolylineResult & { vertices: Int32Array; weights: Float64Array; length: number }
  'geometry.exactDistance'(surface: Surface, vStart: number, vEnd: number): number
  /** exact distance [nV] and log map [2 nV, row-major] from a vertex (index) or a face point (barycentrics in the face's vertex order) to every vertex */
  'geometry.exactFromPoint'(surface: Surface, source: number | { face: number; bary: ArrayLike<number> }): { distance: Float64Array; log: Float64Array }
  'geometry.parallel'(surface: Surface, sourceVerts: Ints, sourceVectors: Floats, tCoef?: number): Float64Array
  'geometry.parallel'(surface: Surface, sourceVerts: Ints[], sourceVectors: Floats[], tCoef?: number): Float64Array[]
  'geometry.extendScalar'(surface: Surface, sourceVerts: Ints, sourceValues: Floats, tCoef?: number): Float64Array
  'geometry.logMap'(surface: Surface, sourceVertex: number, strategy?: LogMapStrategy, tCoef?: number): LogMapResult
  'geometry.logMap'(surface: Surface, sourceVertices: Ints, strategy?: LogMapStrategy, tCoef?: number): LogMapResult[]
  'geometry.findCenter'(surface: Surface, sourceVerts: Ints, p?: number, tCoef?: number): Float64Array
  'geometry.signedHeat'(surface: Surface, curve: Ints, isLoop?: boolean, levelSet?: SignedHeatLevelSet, tCoef?: number): Float64Array
  'geometry.signedHeat'(surface: Surface, curves: Ints[], isLoop?: boolean, levelSet?: SignedHeatLevelSet, tCoef?: number): Float64Array[]
  'geometry.smoothFace'(surface: Surface, nSym?: number, alignToCurvature?: boolean): Float64Array
  'geometry.smoothVertex'(surface: Surface, nSym?: number, alignToCurvature?: boolean): SmoothVertexFieldResult
  'synthesis.delta'(nV: number, sourceVerts: Ints, sourceValues: Floats): Float64Array
  /** M: the mass (COO); eig: M-orthonormal modes (an eigensolve with normalize); u0: [nV] */
  'synthesis.heatDiffusion'(M: SparseMatrixCOO, eig: EigenResult, u0: Float64Array, timesteps: Floats, alpha?: number): TimeSeriesField
  'synthesis.dampedWave'(eig: EigenResult, modeIndices: Ints, amplitudes: Floats, dampings: Floats, phases: Floats, timesteps: Floats): TimeSeriesField
  'synthesis.randomDecomposed1Form'(surface: Surface, alphaStrength: number, betaStrength: number, gammaStrength: number, seed?: number): Float64Array

  /** Six-group nested namespace `nxr.manifold.*` over the surface data (made once). */
  createManifoldContext(
    vertices: Float64Array | number[],
    faces:    Int32Array  | number[],
    options?: SurfaceOptions,
  ): ManifoldContext
  /** Functional form of the six-group namespace. Each leaf takes a
   *  ManifoldContext as its first argument. Identical behaviour to the
   *  per-context methods. */
  nxr: {
    manifold: ManifoldFunctional
  }
}

/* ---------------------------------------------------------------------- */
/*           Six-group nested namespace `nxr.manifold.*` types            */
/* ---------------------------------------------------------------------- */

/** Solver entry points — Poisson, heat diffusion, eigenmodes, Hodge. */
export interface SolveGroup {
  poisson(
    sourceVerts:  Int32Array  | number[],
    sourceValues: Float64Array | number[],
  ): Float64Array
  /** Heat diffusion over the given timesteps; consolidates the legacy
   *  `heatDiffusion` entry point. */
  heat(
    sources:      Int32Array  | number[],
    sourceValues: Float64Array | number[],
    timesteps:    Float64Array | number[],
    alpha?: number,
  ): TimeSeriesField
  /** sigma omitted/NaN = the shift ladder (Node: always the ladder); an explicit sigma is a fixed shift (wasm). */
  eigen(k: number, sigma?: number): EigenResult
  /** Provisional placement — Hodge–Helmholtz decomposition may be
   *  reclassified to `operator.hodgeDecomp` in a future round. */
  hodge(omega: Float64Array): HodgeResult
}

/** DEC and mesh operators as individual sparse matrices. The first call
 *  to any DEC piece triggers a single bulk assembly internally; the same
 *  applies to mass / stiffness / laplacian. */
export interface OperatorGroup {
  d0():           SparseMatrixCOO
  d1():           SparseMatrixCOO
  star0():        SparseMatrixCOO
  star1():        SparseMatrixCOO
  star2():        SparseMatrixCOO
  star1Inverse(): SparseMatrixCOO
  mass():         SparseMatrixCOO
  stiffness():    SparseMatrixCOO
  /** Cotangent Laplacian — same matrix as `stiffness()`. */
  laplacian():    SparseMatrixCOO
  /** Connection Laplacian on the chosen domain. Drives smoothest n-RoSy
   *  direction fields, parallel-transport energies, and tangent-bundle
   *  eigendecompositions. Result-level cached by all four options. */
  connectionLaplacian(options?: ConnectionLaplacianOptions): ConnectionLaplacianResult
  /** Drop the cached DEC and mesh operator results so the next access
   *  re-runs `assembleDECOperators` / `assembleManifoldOperators`. */
  invalidateCache(): void
}

/** Geometric selections. Most are round-1 placeholders. */
export interface QueryGroup {
  vertex(v: number): { vertexIndex: number }
  /** TODO (round-1 stub) — flip-out geodesic between two vertices. */
  line(v1: number, v2: number):    { method: 'todo'; name: string }
  /** TODO (round-1 stub) — isoline of geodesic distance at radius r. */
  circle(v: number, r: number):    { method: 'todo'; name: string }
  /** TODO (round-1 stub) — vertex set within geodesic radius r. */
  region(v: number, r: number):    { method: 'todo'; name: string }
  /** Single-level isoline of a per-vertex scalar field. */
  isoline(field: Float64Array | number[], level: number): SegmentsResult
  /** Karcher mean of source vertices (vector-heat findCenter). */
  center(sourceVerts: Int32Array | number[], p?: number): Float64Array
}

/** `measure.distance` is callable (heat-method geodesic) AND carries a
 *  `.signed` sub-leaf for the signed heat distance from a curve. */
export interface DistanceFn {
  (sourceVerts: Int32Array | number[]): Float64Array
  signed(
    curveVerts: Int32Array | number[],
    isLoop?:    boolean,
    levelSet?:  SignedHeatLevelSet,
  ): Float64Array
}

export interface MeasureGroup {
  distance: DistanceFn
  /** TODO (round-1 stub) — area of a region selection. */
  area(region: unknown):                    { method: 'todo'; name: string }
  /** TODO (round-1 stub) — average a scalar field over a region. */
  density(region: unknown, field: unknown): { method: 'todo'; name: string }
  curvature():                              CurvatureResult
  normal(type?: NormalType):                Float64Array
  frame():                                  FaceFrames
}

export interface UvGroup {
  /** Boundary First Flattening — throws on closed meshes. */
  bff(): Float64Array
  logMap(sourceVertex: number, strategy?: LogMapStrategy): LogMapResult
}

export interface InterpolateGroup {
  transport(
    sourceVerts:   Int32Array  | number[],
    sourceVectors: Float64Array | number[],
  ): Float64Array
  extend(
    sourceVerts:  Int32Array  | number[],
    sourceValues: Float64Array | number[],
  ): Float64Array
  trivial(
    singVerts:  Int32Array  | number[],
    singValues: Float64Array | number[],
  ): DirectionFieldResult
  smoothFace(nSym?: number, alignToCurvature?: boolean): Float64Array
  smoothVertex(nSym?: number, alignToCurvature?: boolean): SmoothVertexFieldResult
}

/** The context of the six-group nested namespace: a plain JS value holding the
 *  surface data (`surface`). It owns no WASM memory; nothing is kept between calls. */
export interface ManifoldContext {
  nV(): number
  nE(): number
  nF(): number

  solve:       SolveGroup
  operator:    OperatorGroup
  query:       QueryGroup
  measure:     MeasureGroup
  uv:          UvGroup
  interpolate: InterpolateGroup

  /** The surface data (`geometry.surfaceArrays`): pass it to any flat function. */
  surface: Surface
  /** Nothing to free (the surface is a JS value); kept for API compatibility. */
  dispose(): void
}

/** Functional form of the six-group namespace. Each leaf takes the
 *  ManifoldContext as its first argument and forwards to the matching
 *  per-context method. */
export interface ManifoldFunctional {
  solve: {
    poisson(mctx: ManifoldContext, sourceVerts: Int32Array | number[], sourceValues: Float64Array | number[]): Float64Array
    heat(
      mctx: ManifoldContext,
      sources: Int32Array | number[],
      sourceValues: Float64Array | number[],
      timesteps: Float64Array | number[],
      alpha?: number,
    ): TimeSeriesField
    eigen(mctx: ManifoldContext, k: number, sigma?: number): EigenResult
    hodge(mctx: ManifoldContext, omega: Float64Array): HodgeResult
  }
  operator: {
    d0(mctx: ManifoldContext):           SparseMatrixCOO
    d1(mctx: ManifoldContext):           SparseMatrixCOO
    star0(mctx: ManifoldContext):        SparseMatrixCOO
    star1(mctx: ManifoldContext):        SparseMatrixCOO
    star2(mctx: ManifoldContext):        SparseMatrixCOO
    star1Inverse(mctx: ManifoldContext): SparseMatrixCOO
    mass(mctx: ManifoldContext):         SparseMatrixCOO
    stiffness(mctx: ManifoldContext):    SparseMatrixCOO
    laplacian(mctx: ManifoldContext):    SparseMatrixCOO
    connectionLaplacian(
      mctx:    ManifoldContext,
      options?: ConnectionLaplacianOptions,
    ): ConnectionLaplacianResult
  }
  query: {
    vertex(mctx: ManifoldContext, v: number):                    { vertexIndex: number }
    line(mctx: ManifoldContext, v1: number, v2: number):         { method: 'todo'; name: string }
    circle(mctx: ManifoldContext, v: number, r: number):         { method: 'todo'; name: string }
    region(mctx: ManifoldContext, v: number, r: number):         { method: 'todo'; name: string }
    isoline(mctx: ManifoldContext, field: Float64Array | number[], level: number): SegmentsResult
    center(mctx: ManifoldContext, sourceVerts: Int32Array | number[], p?: number): Float64Array
  }
  measure: {
    distance: {
      (mctx: ManifoldContext, sourceVerts: Int32Array | number[]): Float64Array
      signed(
        mctx: ManifoldContext,
        curveVerts: Int32Array | number[],
        isLoop?: boolean,
        levelSet?: SignedHeatLevelSet,
      ): Float64Array
    }
    area(mctx: ManifoldContext, region: unknown):                    { method: 'todo'; name: string }
    density(mctx: ManifoldContext, region: unknown, field: unknown): { method: 'todo'; name: string }
    curvature(mctx: ManifoldContext):                                CurvatureResult
    normal(mctx: ManifoldContext, type?: NormalType):                Float64Array
    frame(mctx: ManifoldContext):                                    FaceFrames
  }
  uv: {
    bff(mctx: ManifoldContext): Float64Array
    logMap(mctx: ManifoldContext, sourceVertex: number, strategy?: LogMapStrategy): LogMapResult
  }
  interpolate: {
    transport(
      mctx: ManifoldContext,
      sourceVerts: Int32Array | number[],
      sourceVectors: Float64Array | number[],
    ): Float64Array
    extend(
      mctx: ManifoldContext,
      sourceVerts: Int32Array | number[],
      sourceValues: Float64Array | number[],
    ): Float64Array
    trivial(
      mctx: ManifoldContext,
      singVerts: Int32Array | number[],
      singValues: Float64Array | number[],
    ): DirectionFieldResult
    smoothFace(mctx: ManifoldContext, nSym?: number, alignToCurvature?: boolean): Float64Array
    smoothVertex(mctx: ManifoldContext, nSym?: number, alignToCurvature?: boolean): SmoothVertexFieldResult
  }
}

export interface InitOptions {
  /** Custom locator for nxr_compute.wasm. */
  locateFile?: (filename: string) => string
}

/**
 * Load the nxr-compute WASM module. Idempotent. The promise resolves to a
 * NxrCompute object exposing the public API.
 */
export function initNxrCompute(options?: InitOptions): Promise<NxrCompute>

export default initNxrCompute
