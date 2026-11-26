// Smoke test for the nxr-compute WASM binding. Loads native/build_wasm/nxr_compute.js,
// builds the same icosahedron fixture used by the native C++ tests,
// runs the precompute pipeline, and asserts shapes / values match the
// expected results. Verifies the entire pipeline:
//   JS → WASM heap → Embind → nxr-compute C++ → Eigen → return → JS
//
// Usage (from repo root):
//   node scripts/_smoke-wasm.mjs

import path from 'node:path'
import { pathToFileURL, fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot  = path.resolve(__dirname, '..')

// the module to test: argv[2] (the unified nxr-neuro module, bindings/wasm-io's nxr_io.mjs, has the same toolbox functions), else the standalone compute build
const wasmJsPath = process.argv[2] ? path.resolve(process.argv[2]) : path.join(repoRoot, 'build_wasm', 'nxr_compute.js')
const moduleUrl  = pathToFileURL(wasmJsPath).href

console.log(`[smoke] loading ${wasmJsPath}`)
const { default: createNxrComputeModule } = await import(moduleUrl)
const nxrCompute = await createNxrComputeModule()
// the WRAPPER over the module under test: given a module (argv[2]), bindings/wasm/js/wrap.mjs over THAT instance; else the standalone index.mjs's
const wrappedModule = async () => process.argv[2]
  ? (await import('../wasm/js/wrap.mjs')).wrap(nxrCompute)
  : (await import('../wasm/js/index.mjs')).initNxrCompute()
console.log(`[smoke] module ready — version: ${nxrCompute.version()}`)

// ── Icosahedron fixture (matches test_eigen.cpp) ──────────────
const t = (1 + Math.sqrt(5)) / 2
const raw = [
  -1, t, 0,   1, t, 0,  -1, -t, 0,   1, -t, 0,
   0, -1, t,  0, 1, t,   0, -1, -t,  0, 1, -t,
   t, 0, -1,  t, 0, 1,  -t, 0, -1,  -t, 0, 1,
]
const verts = new Float64Array(36)
for (let i = 0; i < 36; i += 3) {
  const len = Math.hypot(raw[i], raw[i + 1], raw[i + 2])
  verts[i]     = raw[i]     / len
  verts[i + 1] = raw[i + 1] / len
  verts[i + 2] = raw[i + 2] / len
}
const faces = new Int32Array([
  0,11, 5,  0, 5, 1,  0, 1, 7,  0, 7,10,  0,10,11,
  1, 5, 9,  5,11, 4, 11,10, 2, 10, 7, 6,  7, 1, 8,
  3, 9, 4,  3, 4, 2,  3, 2, 6,  3, 6, 8,  3, 8, 9,
  4, 9, 5,  2, 4,11,  6, 2,10,  8, 6, 7,  9, 8, 1,
])

// ── Test 1: the surface as data (made once) and its sizes ───────────────────────
const S = nxrCompute['geometry.surfaceArrays'](verts, faces, null)
const ops0 = nxrCompute['operators.assembleManifoldOperators'](S, '')
const nV = ops0.nV, nF = ops0.nF, nE = ops0.nE
console.log(`[smoke] surface: nV=${nV}, nF=${nF}, nE=${nE}`)

function require(cond, msg) {
  if (!cond) {
    console.error('FAIL:', msg)
    process.exit(1)
  }
}
require(nV === 12, 'icosahedron has 12 vertices')
require(nF === 20, 'icosahedron has 20 faces')
require(nE === 30, 'icosahedron has 30 edges')

// ── Test 2: precompute (the visualization-defaults pack) ──────
const t0 = performance.now()
const Kc = ops0.cotanLaplacian, Mc = ops0.mass
const eigen = (K, M, k, sigma, normalize, removeDC) =>
  nxrCompute['operators.eigen'](K.row, K.col, K.data, M.row, M.col, M.data, K.rows, k, sigma, 0, 0, 0, normalize, removeDC)
const data = {
  operators: ops0,
  dec: nxrCompute['operators.assembleDECOperators'](S),
  eigenmodes: eigen(Kc, Mc, 6, -1e-8, true, true),   // k=6 modes, normalized, DC removed
  frames: nxrCompute['geometry.frames'](S),
}
const elapsedPrecompute = (performance.now() - t0).toFixed(2)
console.log(`[smoke] assemble + eigen(k: 6) + frames: ${elapsedPrecompute} ms`)

// Operators
require(data.operators.nV === 12, 'operators.nV')
require(Math.abs(data.operators.totalArea - 9.5745) < 1e-3,
  `unit-sphere icosahedron area should be ≈ 9.5745, got ${data.operators.totalArea}`)
require(data.operators.vertexNormals instanceof Float64Array, 'vertexNormals is Float64Array')
require(data.operators.vertexNormals.length === 12 * 3, 'vertexNormals length nV*3')
require(data.operators.cotanLaplacian.rows === 12, 'cotanLaplacian rows = nV')
require(data.operators.cotanLaplacian.cols === 12, 'cotanLaplacian cols = nV')
require(data.operators.cotanLaplacian.row instanceof Int32Array, 'cotanLaplacian COO.row Int32Array')
console.log(`  operators ✓ (totalArea=${data.operators.totalArea.toFixed(4)}, ` +
            `cotanLaplacian nnz=${data.operators.cotanLaplacian.nnz})`)

// DEC
require(data.dec.d0.rows === 30, 'd0 rows = nE')
require(data.dec.d0.cols === 12, 'd0 cols = nV')
require(data.dec.d1.rows === 20, 'd1 rows = nF')
require(data.dec.d1.cols === 30, 'd1 cols = nE')
console.log(`  DEC ✓ (d0 ${data.dec.d0.rows}×${data.dec.d0.cols}, ` +
            `d1 ${data.dec.d1.rows}×${data.dec.d1.cols})`)

// Eigenmodes — after removeDC, k should be 5 (one DC mode dropped)
require(data.eigenmodes.k === 5, `expected k=5 (post-removeDC), got ${data.eigenmodes.k}`)
require(data.eigenmodes.eigenvalues instanceof Float64Array, 'eigenvalues Float64Array')
require(data.eigenmodes.eigenvalues.length === 5, 'eigenvalues length')
require(data.eigenmodes.eigenvectors.length === 12 * 5, 'eigenvectors length nV*K')
const evals = Array.from(data.eigenmodes.eigenvalues)
console.log(`  eigenmodes ✓ k=${data.eigenmodes.k}, λ=[${evals.map(v => v.toFixed(4)).join(', ')}]`)

// Icosahedron canonical spectrum: triplet at λ=2, doublet at λ≈4.34
require(Math.abs(evals[0] - 2.0) < 1e-6,
  `expected λ_0 ≈ 2.0 on icosahedron, got ${evals[0]}`)
require(Math.abs(evals[3] - 4.34164) < 1e-3,
  `expected λ_3 ≈ 4.34, got ${evals[3]}`)

// Face frames
require(data.frames.e1.length      === 20 * 3, 'face frames e1 [F*3]')
require(data.frames.e2.length      === 20 * 3, 'face frames e2 [F*3]')
require(data.frames.normals.length === 20 * 3, 'face frames normals [F*3]')

// Verify orthonormality: e1·e2 ≈ 0, |e1| ≈ 1, e1×e2 ≈ n
let maxOrth = 0, maxNorm = 0
for (let f = 0; f < 20; f++) {
  const e1x = data.frames.e1[f*3], e1y = data.frames.e1[f*3+1], e1z = data.frames.e1[f*3+2]
  const e2x = data.frames.e2[f*3], e2y = data.frames.e2[f*3+1], e2z = data.frames.e2[f*3+2]
  maxOrth = Math.max(maxOrth, Math.abs(e1x*e2x + e1y*e2y + e1z*e2z))
  maxNorm = Math.max(maxNorm, Math.abs(Math.hypot(e1x, e1y, e1z) - 1))
}
require(maxOrth < 1e-12, `face frames orthogonal (max|e1·e2|=${maxOrth})`)
require(maxNorm < 1e-12, `face frames unit-length (max|‖e1‖−1|=${maxNorm})`)
console.log(`  frames ✓ (orthonormal to ${maxOrth.toExponential(2)})`)

// ── Test 3: M-orthonormality of eigenvectors ──────────────────
//
// U' M U should be ≈ identity. Phase A standardised vMajor
// (row-major) layout for [V × K] eigenvectors: U[v*K + k] gives
// vertex v's value in mode k. Mode k extraction is therefore a
// strided read (stride K), matching the cortical-flow Zarr schema
// `manifold/eigenmodes/eigenvectors` shape [V, K].
const evecs = data.eigenmodes.eigenvectors  // [V*K] row-major (vMajor)
const K = data.eigenmodes.k
const massCOO = data.operators.mass

const massDiag = new Float64Array(nV)
for (let i = 0; i < massCOO.row.length; i++) {
  if (massCOO.row[i] === massCOO.col[i]) {
    massDiag[massCOO.row[i]] = massCOO.data[i]
  }
}

// Mode extraction with vMajor: stride K read.
function extractMode(k) {
  const out = new Float64Array(nV)
  for (let v = 0; v < nV; v++) out[v] = evecs[v * K + k]
  return out
}

let maxOffDiag = 0
let maxDiagErr = 0
for (let j = 0; j < K; j++) {
  const u_j = extractMode(j)
  for (let k = 0; k < K; k++) {
    const u_k = extractMode(k)
    let dot = 0
    for (let v = 0; v < nV; v++) {
      dot += u_j[v] * massDiag[v] * u_k[v]
    }
    if (j === k) maxDiagErr  = Math.max(maxDiagErr,  Math.abs(dot - 1))
    else         maxOffDiag  = Math.max(maxOffDiag,  Math.abs(dot))
  }
}
require(maxOffDiag < 1e-10, `M-orth: max|U'MU|_off = ${maxOffDiag}`)
require(maxDiagErr < 1e-10, `M-orth: max|U'MU - I|_diag = ${maxDiagErr}`)
console.log(`  M-orthonormality ✓ (off-diag max=${maxOffDiag.toExponential(2)}, ` +
            `diag err max=${maxDiagErr.toExponential(2)})`)

// ── Test 4: face-centered scalar gradient ─────────────────────
const u_t = new Float64Array(nV).fill(0)
u_t[0] = 1.0       // delta at vertex 0
const grad = nxrCompute['operators.gradient'](S, u_t)
require(grad.length === nF * 3, 'gradient length nF*3')
let maxGradMag = 0
for (let f = 0; f < nF; f++) {
  const m = Math.hypot(grad[f*3], grad[f*3+1], grad[f*3+2])
  maxGradMag = Math.max(maxGradMag, m)
}
require(maxGradMag > 0, 'gradient nontrivial')
console.log(`  gradient ✓ (max |∇u| = ${maxGradMag.toFixed(4)})`)

// ── Test 5: geodesic distance from a vertex ───────────────────
const dists = nxrCompute['geometry.heat'](S, new Int32Array([0]), 1.0, 'intrinsic')
require(dists.length === nV, 'geodesic distances length nV')
require(Math.abs(dists[0]) < 1e-9, `distance at source = 0, got ${dists[0]}`)
let maxDist = 0
for (const d of dists) if (d > maxDist) maxDist = d
console.log(`  heat ✓ (max d = ${maxDist.toFixed(4)})`)
{
  // the batch form: one factorization, an Array of fields
  const both = nxrCompute['geometry.heat'](S, [new Int32Array([0]), new Int32Array([3])], 1.0, 'intrinsic')
  require(Array.isArray(both) && both.length === 2 && both[0].length === nV, 'heat batch: an Array of two fields')
  require(both[0].every((x, i) => Math.abs(x - dists[i]) < 1e-9), 'heat batch: the first field is the single-set field')
}

// ── Test 6: geodesic path between antipodes ───────────────────
const path0 = nxrCompute['geometry.tracePath'](S, 0, 3)
require(path0.nPoints >= 2, 'path has at least 2 points')
require(path0.positions.length === path0.nPoints * 3, 'path positions [N*3]')
let pathLen = 0
for (let i = 0; i + 1 < path0.nPoints; i++) {
  const dx = path0.positions[(i+1)*3]   - path0.positions[i*3]
  const dy = path0.positions[(i+1)*3+1] - path0.positions[i*3+1]
  const dz = path0.positions[(i+1)*3+2] - path0.positions[i*3+2]
  pathLen += Math.hypot(dx, dy, dz)
}
require(pathLen > 2.0 && pathLen < Math.PI,
  `discrete antipode path in (chord=2.0, sphere-arc=π), got ${pathLen}`)
console.log(`  tracePath ✓ (antipode path length ${pathLen.toFixed(4)})`)

// ── Test 6b: the exact (MMP) distance the path's length bounds ─
const exact03 = nxrCompute['geometry.exactDistance'](S, 0, 3)
require(exact03 > 2.0 && exact03 <= path0.length + 1e-12,
  `exact antipode distance in (chord=2.0, flip-out length ${path0.length}], got ${exact03}`)
require(nxrCompute['geometry.exactDistance'](S, 0, 0) === 0, 'exact distance to itself 0')
console.log(`  exactDistance ✓ (antipodes ${exact03.toFixed(4)} <= flip-out ${path0.length.toFixed(4)})`)

// ── Test 7: heat diffusion (spectral generator) ───────────────
const ts = Float64Array.from({ length: 5 }, (_, i) => i * 0.1)
const u0 = nxrCompute['synthesis.delta'](nV, new Int32Array([0]), new Float64Array([1.0]))
const heat = nxrCompute['synthesis.heatDiffusion'](
  Mc,
  data.eigenmodes,
  u0,
  ts,
  /* alpha = */ 1.0,
)
require(heat.T  === 5, `heat T=5, got ${heat.T}`)
require(heat.nV === 12, `heat nV=12, got ${heat.nV}`)
require(heat.data instanceof Float32Array, 'heat data Float32Array')
require(heat.data.length === 5 * 12, 'heat data length T*V')
console.log(`  heatDiffusion ✓ (T=${heat.T}, nV=${heat.nV})`)

// ── Test 8: vector heat method ───────────────────────────────
{
  const transported = nxrCompute['geometry.parallel'](S,
    new Int32Array([0]),
    new Float64Array([1.0, 0.0, 0.0]),
    1.0,
  )
  require(transported.length === nV * 3, 'vhm transport length nV*3')
  let maxNorm = 0
  for (let v = 0; v < nV; v++) {
    maxNorm = Math.max(maxNorm,
      Math.hypot(transported[v*3], transported[v*3+1], transported[v*3+2]))
  }
  require(maxNorm > 1e-6, 'vhm transport produced non-trivial vectors')

  const extended = nxrCompute['geometry.extendScalar'](S,
    new Int32Array([0, 6]),
    new Float64Array([1.0, 0.0]),
    1.0,
  )
  require(extended.length === nV, 'vhm extendScalar length nV')

  const logmap = nxrCompute['geometry.logMap'](S, 0, /* AffineLocal */ 1, 1.0)
  require(logmap.logCoords.length === nV * 2, 'logmap V*2')
  require(logmap.sourceE1.length === 3 && logmap.sourceE2.length === 3, 'logmap source frame 3')
  require(Math.hypot(logmap.logCoords[0], logmap.logCoords[1]) < 1e-6, 'logmap zero at source')

  const center = nxrCompute['geometry.findCenter'](S, new Int32Array([0, 1, 5]), 2, 1.0)
  require(center.length === 3 && center.every(Number.isFinite), 'findCenter xyz finite')
  console.log(`  vectorHeat ✓ (transport max=${maxNorm.toFixed(3)}, ` +
              `center=[${[...center].map(v => v.toFixed(3)).join(', ')}])`)
}

// ── Test 9: signed heat method ───────────────────────────────
{
  const sd = nxrCompute['geometry.signedHeat'](S,
    new Int32Array([11, 5, 1, 7, 10]),
    /* isLoop = */ true,
    /* ZeroSet = */ 1,
    1.0,
  )
  require(sd.length === nV, 'signed distance length nV')
  let mn = Infinity, mx = -Infinity
  for (const d of sd) { if (d < mn) mn = d; if (d > mx) mx = d }
  require(mn < 0 && mx > 0, `signed range straddles zero, got [${mn}, ${mx}]`)
  console.log(`  signedHeat ✓ (range [${mn.toFixed(3)}, ${mx.toFixed(3)}])`)
}

// ── Test 10: smooth NRoSy direction fields ──────────────────
{
  const faceField = nxrCompute['geometry.smoothFace'](S, 4, false)
  require(faceField.length === nF * 3, 'smooth face field length F*3')

  const vfield = nxrCompute['geometry.smoothVertex'](S, 2, false)
  require(vfield.vertexVectors.length  === nV * 3, 'smooth vertex field V*3')
  require(vfield.vertexFieldRaw.length === nV * 2, 'smooth vertex field raw V*2')
  require(vfield.nSym === 2, 'nSym preserved')
  console.log(`  smoothField ✓ (face nF=${nF}, vertex nV=${nV})`)

}

// ── Test 12: high-level wrapper supplies defaults ───────────
//
// The raw Embind functions above require every argument. The
// index.mjs wrapper supplies the optional-argument defaults
// declared in index.d.ts. Verify the wrapper is loadable AND
// that calling each new method without its optional args works.
{
  const wrapped = await wrappedModule()                  // idempotent — re-uses the loaded module
  const wS = wrapped['geometry.surfaceArrays'](verts, faces)

  // Defaults: strategy=AffineLocal, p=2, isLoop=true, levelSet=ZeroSet,
  //           nSym=4 (face) / 2 (vertex), connectOnSingularities=true.
  const lm = wrapped['geometry.logMap'](wS, 0)
  require(lm.logCoords.length === nV * 2, 'wrapped logMap default strategy works')
  const c  = wrapped['geometry.findCenter'](wS, [0, 1, 5])
  require(c.length === 3, 'wrapped findCenter default p works')
  const sd = wrapped['geometry.signedHeat'](wS, [11, 5, 1, 7, 10])
  require(sd.length === nV, 'wrapped signedHeat default flags work')
  const ff = wrapped['geometry.smoothFace'](wS)
  require(ff.length === nF * 3, 'wrapped smoothFace default nSym works')
  const vf = wrapped['geometry.smoothVertex'](wS)
  require(vf.vertexFieldRaw.length === nV * 2, 'wrapped smoothVertex default nSym works')
  console.log('  index.mjs defaults ✓')
}

// ── Test 13: the connection Laplacian, and the old names refused ──────
{
  const isCOO = (m, r, c) =>
    m && m.rows === r && m.cols === c &&
    m.row instanceof Int32Array && m.data instanceof Float64Array && m.nnz > 0

  const cl = nxrCompute['operators.assembleConnectionLaplacian'](S, { format: 'complex' })
  require(cl.K.rows === nV && cl.K.realData instanceof Float64Array &&
          cl.K.imagData instanceof Float64Array, 'the connection Laplacian is complex COO')
  const dec = data.dec
  require(isCOO(dec.d0, nE, nV) && isCOO(dec.d1, nF, nE), 'assembleDECOperators returns d0 and d1')

  // the wrapper refuses an old name with the new one (no alias)
  const wrapped = await wrappedModule()
  for (const [old, now] of [['heat', 'geometry.heat'], ['solve', 'operators.eigen'], ['hodge', 'signal.hodge']]) {
    let msg = ''
    try { wrapped[old]() } catch (e) { msg = e.message }
    require(msg.includes(`'${old}' was renamed '${now}'`), `'${old}' is refused naming '${now}'`)
  }
  let removed = ''
  try { wrapped.createContext() } catch (e) { removed = e.message }
  require(removed.includes('geometry.surfaceArrays'), 'createContext is refused: the surface is data now')

  // Error paths surface as JS Errors with the "[CODE] ..." message shape.
  let threwSurface = false
  try { nxrCompute['geometry.heat']({}, new Int32Array([0]), 1.0, 'intrinsic') } catch { threwSurface = true }
  require(threwSurface, 'a surface without its arrays throws')
  console.log('  connection Laplacian and refused names ✓')
}

// ── Test 14: named-operator eigensolve (Dirac, no JS-side ⊗I₄) ───
//
// One call assembles K + its natural mass C++-side and eigensolves — the
// consumer never builds B = M⊗I₄. multiplets:true gives exact 4-fold multiplets;
// dense:true cross-checks on this small mesh.
{
  // Scalar Laplace–Beltrami via the same entry point.
  const lb = nxrCompute['operators.eigenOperator'](S, { operator: 'laplacian', subtype: 'cotan', k: 6 })
  require(lb.eigenvalues.length === 6 && lb.blockSize === 1, 'eigs(cotan) k=6, blockSize 1')
  require(Math.abs(lb.eigenvalues[0]) < 1e-6, 'cotan lowest eigenvalue ≈ 0 (constant mode)')

  // Dirac eigensolve — eigenvectors are [4·nV × k] vMajor.
  const d = nxrCompute['operators.eigenOperator'](S, { operator: 'dirac', tau: 0.5, k: 8 })
  require(d.blockSize === 4, 'eigs(dirac) blockSize 4')
  require(d.eigenvalues.length === d.k, 'eigenvalues length == k')
  require(d.eigenvectors.length === 4 * nV * d.k, 'eigenvectors length 4·nV·k')

  // Exact 4-fold multiplets via ℍ-reconstruction.
  const dm = nxrCompute['operators.eigenOperator'](S, { operator: 'dirac', tau: 0.5, k: 8, multiplets: true })
  require(dm.k % 4 === 0, 'multiplets: k rounded to a multiple of 4')
  const ev = Array.from(dm.eigenvalues)
  let maxSpread = 0
  for (let g = 0; 4*g + 3 < ev.length; g++) {
    const seg = ev.slice(4*g, 4*g + 4)
    maxSpread = Math.max(maxSpread, (Math.max(...seg) - Math.min(...seg)) / (1 + Math.abs(seg[3])))
  }
  require(maxSpread < 1e-8, `multiplets are exact 4-fold (spread ${maxSpread.toExponential(2)})`)

  // Dense exact solve agrees on the lowest 8.
  const dd = nxrCompute['operators.eigenOperator'](S, { operator: 'dirac', tau: 0.5, k: 8, dense: true })
  let maxDiff = 0
  for (let i = 0; i < 8; i++) maxDiff = Math.max(maxDiff, Math.abs(dm.eigenvalues[i] - dd.eigenvalues[i]))
  require(maxDiff < 1e-6, `sparse-reconstruct == dense (lowest 8, Δ=${maxDiff.toExponential(2)})`)
  require(Math.abs(dd.eigenvalues[0]) < 1e-8 && Math.abs(dd.eigenvalues[3]) < 1e-8,
          'dirac has a 4-fold zero (constant mode)')

  // diracFace eigensolve (closed mesh).
  const df = nxrCompute['operators.eigenOperator'](S, { operator: 'diracFace', tau: 0.5, k: 8, multiplets: true })
  require(df.k % 4 === 0 && df.eigenvectors.length === 4 * nF * df.k, 'eigs(diracFace) shapes')

  // Missing k → throws.
  let threw = false
  try { nxrCompute['operators.eigenOperator'](S, { operator: 'dirac', tau: 0.5 }) } catch { threw = true }
  require(threw, 'eigs without k throws')
  console.log(`  eigs ✓ (dirac k=${dm.k} multiplets, dense Δ=${maxDiff.toExponential(2)}, ` +
              `cotan λ₁=${lb.eigenvalues[1].toFixed(4)})`)
}

console.log('[smoke] all assertions passed ✓')
