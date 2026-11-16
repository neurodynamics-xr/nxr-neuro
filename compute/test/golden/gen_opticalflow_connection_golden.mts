// Writes opticalflow_connection_golden.txt: the Studio's own optical flow (flow/opticalFlow.ts: componentPencil, intrinsicGradient, opticalFlow) on small meshes, for
// test_opticalflow_connection.cpp. Per mesh: positions, faces, the maps (float32 values, as the Studio takes them), alpha, the pencil's C (triplets of the real 2n x 2n matrix), M,
// the vertex normals, the gradient in C's gauge of frame 0 (normalised), the gauge solution z of frame 0 (a dense float64 solve of the Studio's system: the Studio's own
// solver returns only the embedded float32 velocity) and the Studio's velocity (float32, frame-major).
// Run:  NXR_CORTICAL_FLOW=<path to nxr-cortical-flow> $NXR_CORTICAL_FLOW/node_modules/.bin/tsx compute/test/golden/gen_opticalflow_connection_golden.mts compute/test/golden/opticalflow_connection_golden.txt
import { writeFileSync } from 'node:fs'
const root = process.env.NXR_CORTICAL_FLOW
if (!root) throw new Error('set NXR_CORTICAL_FLOW to the Studio monorepo checkout')
const { opticalFlow, componentPencil, intrinsicGradient, embedAt } = await import(`${root}/studio/src/engine/toolbox/flow/opticalFlow.ts`)

/* the Studio test's sphere (opticalFlow.test.ts): an icosahedron subdivided `sub` times onto the sphere of radius R */
function sphere(sub: number, R = 0.1) {
  const t = (1 + Math.sqrt(5)) / 2
  const verts: number[][] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
  let tris = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
  for (let s = 0; s < sub; s++) {
    const mid = new Map<string, number>()
    const mp = (a: number, b: number) => { const k = a < b ? `${a}_${b}` : `${b}_${a}`; const h = mid.get(k); if (h !== undefined) return h; verts.push([0, 1, 2].map((c) => (verts[a]![c]! + verts[b]![c]!) / 2)); mid.set(k, verts.length - 1); return verts.length - 1 }
    tris = tris.flatMap(([a, b, c]) => { const ab = mp(a!, b!), bc = mp(b!, c!), ca = mp(c!, a!); return [[a!, ab, ca], [b!, bc, ab], [c!, ca, bc], [ab, bc, ca]] })
  }
  const positions = new Float64Array(verts.length * 3)
  verts.forEach((p, i) => { const l = Math.hypot(p[0]!, p[1]!, p[2]!); for (let c = 0; c < 3; c++) positions[3 * i + c] = R * p[c]! / l })
  return { positions, faces: Int32Array.from(tris.flat()), n: verts.length }
}
const f = (a: ArrayLike<number>) => Array.from(a, (x) => Number(x).toPrecision(17)).join(' ')
let out = ''
const emit = (name: string, rows: number, cols: number, a: ArrayLike<number>) => { out += `${name} ${rows} ${cols}\n${f(a)}\n` }

/* a travelling bump: a Gaussian of width sigma whose centre turns about y */
function bump(P: ArrayLike<number>, n: number, nT: number, R: number, omega0 = 3, dt = 0.004) {
  const sigma = 0.2 * R, I = new Float32Array(nT * n)
  for (let k = 0; k < nT; k++) {
    const th = omega0 * k * dt, c = [R * Math.cos(th), 0, -R * Math.sin(th)]
    for (let v = 0; v < n; v++) {
      const l = Math.hypot(P[3 * v]!, P[3 * v + 1]!, P[3 * v + 2]!), cosang = Math.min(1, Math.max(-1, (P[3 * v]! * c[0]! + P[3 * v + 1]! * c[1]! + P[3 * v + 2]! * c[2]!) / (l * R)))
      const d = R * Math.acos(cosang)
      I[k * n + v] = Math.exp(-(d * d) / (2 * sigma * sigma))
    }
  }
  return I
}
let seed = 9; const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5 }

function mesh(name: string, positions: Float64Array, faces: Int32Array, n: number, R: number, alpha: number, nT = 3) {
  const I = bump(positions, n, nT, R)
  let peak = 0; for (const x of I) peak = Math.max(peak, x)
  const { C, M, normals, corners, grad } = componentPencil(positions, faces, 0, n)
  const cur = Float64Array.from({ length: n }, (_, v) => I[v]! / peak)
  const g = intrinsicGradient(corners, grad, cur)
  /* the gauge solution of frame 0 -> 1: the Studio's system written densely (as opticalFlow.test.ts does) and solved in float64 */
  const dI = Float64Array.from({ length: n }, (_, v) => I[n + v]! / peak - cur[v]!)
  const N = 2 * n, A = Array.from({ length: N }, () => new Float64Array(N + 1))
  let dC = 0
  for (let r = 0; r < N; r++) for (let p = C.indptr[r]!; p < C.indptr[r + 1]!; p++) { A[r]![C.indices[p]!]! += alpha * C.data[p]!; if (C.indices[p] === r) dC += C.data[p]! }
  const ridge = 1e-9 * alpha * (dC / N || 1)
  for (let i = 0; i < n; i++) {
    const g1 = g[2 * i]!, g2 = g[2 * i + 1]!, m = M[i]!
    A[2 * i]![2 * i]! += m * g1 * g1 + ridge; A[2 * i + 1]![2 * i + 1]! += m * g2 * g2 + ridge
    A[2 * i]![2 * i + 1]! += m * g1 * g2; A[2 * i + 1]![2 * i]! += m * g1 * g2
    A[2 * i]![N] = -m * g1 * dI[i]!; A[2 * i + 1]![N] = -m * g2 * dI[i]!
  }
  for (let c = 0; c < N; c++) { let r = c; for (let i = c + 1; i < N; i++) if (Math.abs(A[i]![c]!) > Math.abs(A[r]![c]!)) r = i; [A[c], A[r]] = [A[r]!, A[c]!]
    for (let i = c + 1; i < N; i++) { const fct = A[i]![c]! / A[c]![c]!; if (fct) for (let k = c; k <= N; k++) A[i]![k]! -= fct * A[c]![k]! } }
  const z = new Float64Array(N); for (let i = N - 1; i >= 0; i--) { let s = A[i]![N]!; for (let k = i + 1; k < N; k++) s -= A[i]![k]! * z[k]!; z[i] = s / A[i]![i]! }
  const emb = new Float64Array(3 * n); for (let i = 0; i < n; i++) embedAt(corners, normals, i, z[2 * i]!, z[2 * i + 1]!, emb, 3 * i)
  const { velocity } = opticalFlow(I, nT, { positions, faces }, { alpha })
  out += `mesh ${name} ${n} ${nT} ${alpha} ${peak.toPrecision(17)}\n`
  emit('positions', n, 3, positions); emit('faces', faces.length / 3, 3, faces); emit('I', nT, n, I)
  const trip: number[] = []
  for (let r = 0; r < C.n; r++) for (let p = C.indptr[r]!; p < C.indptr[r + 1]!; p++) trip.push(r, C.indices[p]!, C.data[p]!)
  emit('C', trip.length / 3, 3, trip); emit('M', n, 1, M); emit('normals', n, 3, normals); emit('g', n, 2, g); emit('z', n, 2, z); emit('zv', n, 3, emb)
  emit('velocity', nT - 1, 3 * n, velocity)
}

const S1 = sphere(1)
mesh('sphere1', S1.positions, S1.faces, S1.n, 0.1, 0.3)
const S2 = sphere(2)
mesh('sphere2', S2.positions, S2.faces, S2.n, 0.1, 0.01)
/* an IRREGULAR mesh: the sphere jittered and stretched, so that intrinsic Delaunay flips occur and the corners are uneven */
const J = sphere(2), JP = Float64Array.from(J.positions)
for (let v = 0; v < J.n; v++) { JP[3 * v] = JP[3 * v]! * 1.6 * (1 + 0.15 * rnd()); JP[3 * v + 1] = JP[3 * v + 1]! * (1 + 0.15 * rnd()); JP[3 * v + 2] = JP[3 * v + 2]! * 0.7 * (1 + 0.15 * rnd()) }
mesh('irregular', JP, J.faces, J.n, 0.1, 0.05)
/* an OPEN patch (a cap of the irregular mesh): boundary fans */
{
  const keep = new Int32Array(J.n).fill(-1); let m = 0
  for (let v = 0; v < J.n; v++) if (JP[3 * v + 2]! > -0.02) keep[v] = m++
  const P = new Float64Array(3 * m), F: number[] = []
  for (let v = 0; v < J.n; v++) if (keep[v]! >= 0) P.set(JP.subarray(3 * v, 3 * v + 3), 3 * keep[v]!)
  for (let q = 0; q < J.faces.length; q += 3) { const a = keep[J.faces[q]!]!, b = keep[J.faces[q + 1]!]!, c = keep[J.faces[q + 2]!]!; if (a >= 0 && b >= 0 && c >= 0) F.push(a, b, c) }
  mesh('patch', P, Int32Array.from(F), m, 0.1, 0.1)
}
writeFileSync(process.argv[2]!, out)
