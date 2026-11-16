// Writes chart_golden.txt: the Studio's own charts (compute/runtime/surfaceKit.ts: distance, transported, logMap, headingOf, disc, gradient; differential/connectionLaplacian.ts: tangentBasis) on small
// meshes, for test_chart.cpp. Per mesh: positions, faces, then per seed: the heat distance, the transported vector (2 a vertex), r, theta, the heading of a fixed direction, the disc.
// Run:  NXR_CORTICAL_FLOW=<path to nxr-cortical-flow> $NXR_CORTICAL_FLOW/node_modules/.bin/tsx compute/test/golden/gen_chart_golden.mts compute/test/golden/chart_golden.txt
import { writeFileSync } from 'node:fs'
const root = process.env.NXR_CORTICAL_FLOW
if (!root) throw new Error('set NXR_CORTICAL_FLOW to the Studio monorepo checkout')
const { SurfaceKit } = await import(`${root}/studio/src/engine/compute/runtime/surfaceKit.ts`)

function sphere(sub: number, R = 1) {
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
const f = (a: ArrayLike<number>) => Array.from(a, (x) => (Number.isFinite(x) ? Number(x).toPrecision(17) : Number.isNaN(x) ? 'nan' : x > 0 ? 'inf' : '-inf')).join(' ')
let out = ''
const emit = (name: string, rows: number, cols: number, a: ArrayLike<number>) => { out += `${name} ${rows} ${cols}\n${f(a)}\n` }
let seed = 9; const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5 }

function mesh(name: string, positions: Float64Array, faces: Int32Array, n: number, seeds: number[], radius: number) {
  const kit = new SurfaceKit({ positions, faces })
  out += `mesh ${name} ${n} ${seeds.length} ${radius}\n`
  emit('positions', n, 3, positions); emit('faces', faces.length / 3, 3, faces)
  const b = kit.clOf(kit.components[0]).basis
  emit('e1', n, 3, b.e1); emit('e2', n, 3, b.e2); emit('nrm', n, 3, b.n)
  const dir = [0.3, -0.5, 0.8]
  for (const v of seeds) {
    const c = kit.componentOf(v), lm = kit.logMap(v), X = kit.transported(c, v - c.from, 0), X2 = kit.transported(c, v - c.from, 0.7)
    out += `seed ${v}\n`
    emit('dist', n, 1, kit.distance(v)); emit('X', n, 2, X); emit('X07', n, 2, X2); emit('r', n, 1, lm.r); emit('theta', n, 1, lm.theta)
    emit('heading', 1, 1, [kit.headingOf('cl', v, dir as [number, number, number])]); emit('disc', 1, 1, [kit.disc(v, radius).length])
    emit('grad', n, 3, kit.gradient(kit.distance(v)))
  }
}
const S = sphere(2)
mesh('sphere', S.positions, S.faces, S.n, [0, 17, 41], 0.9)
const J = sphere(2), JP = Float64Array.from(J.positions)
for (let v = 0; v < J.n; v++) { JP[3 * v] = JP[3 * v]! * 1.6 * (1 + 0.15 * rnd()); JP[3 * v + 1] = JP[3 * v + 1]! * (1 + 0.15 * rnd()); JP[3 * v + 2] = JP[3 * v + 2]! * 0.7 * (1 + 0.15 * rnd()) }
mesh('irregular', JP, J.faces, J.n, [3, 29], 0.8)
{
  const keep = new Int32Array(J.n).fill(-1); let m = 0
  for (let v = 0; v < J.n; v++) if (JP[3 * v + 2]! > -0.2) keep[v] = m++
  const P = new Float64Array(3 * m), F: number[] = []
  for (let v = 0; v < J.n; v++) if (keep[v]! >= 0) P.set(JP.subarray(3 * v, 3 * v + 3), 3 * keep[v]!)
  for (let q = 0; q < J.faces.length; q += 3) { const a = keep[J.faces[q]!]!, b = keep[J.faces[q + 1]!]!, c = keep[J.faces[q + 2]!]!; if (a >= 0 && b >= 0 && c >= 0) F.push(a, b, c) }
  mesh('patch', P, Int32Array.from(F), m, [5, 20], 0.6)
}
writeFileSync(process.argv[2]!, out)
