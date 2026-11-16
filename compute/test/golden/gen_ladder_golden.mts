// Writes ladder_golden.txt: the Studio's own dyadic tile ladder (engine/ladder.ts: surfaceLadder, vertexAreas, weylLambda, wavelengthOf) on small meshes, and the tile summaries nxr
// computes over it (measure/scalars.py map_stats, measure/vectors.py vector_stats, levi_civita, lc_resultant), for test_ladder.cpp. The transport of the Levi-Civita form is the Studio's own
// (compute/runtime/surfaceKit.ts `transported`), read in a canonical-like frame (the signpost basis turned by a smooth angle alpha, so the gauge angle of e1 is alpha).
// Per mesh: positions, faces, the ladder (per level: scalars, labels, tile areas, vertex counts), the barycentric-mass tile centres and the tile adjacency per level, a scalar map, a tangent vector
// field + its frame, and per level (up to LC_LEVELS) the Studio's rho, Omega, and the frame / Levi-Civita sums. A two-piece mesh gives the SUBJECT tree's codes (h 2^D + t).
// Run:  NXR_CORTICAL_FLOW=<path to nxr-cortical-flow> $NXR_CORTICAL_FLOW/node_modules/.bin/tsx compute/test/golden/gen_ladder_golden.mts compute/test/golden/ladder_golden.txt
//       python3 compute/test/golden/check_ladder_nxr.py compute/test/golden/ladder_golden.txt   (the sums against nxr's own functions, run in nxr's environment)
import { writeFileSync } from 'node:fs'
const root = process.env.NXR_CORTICAL_FLOW
if (!root) throw new Error('set NXR_CORTICAL_FLOW to the Studio monorepo checkout')
const { surfaceLadder, vertexAreas, weylLambda, wavelengthOf } = await import(`${root}/studio/src/engine/ladder.ts`)
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
const LC_LEVELS = 3

/* the barycentric-mass tile centres (the Studio's tileCentres with nxr's mass) and the tile adjacency */
function centres(P: ArrayLike<number>, lab: Int32Array, tiles: number, mass: Float64Array) {
  const w = new Float64Array(tiles), c = new Float64Array(3 * tiles)
  for (let i = 0; i < lab.length; i++) { w[lab[i]!]! += mass[i]!; for (let d = 0; d < 3; d++) c[3 * lab[i]! + d]! += mass[i]! * P[3 * i + d]! }
  const best = new Int32Array(tiles).fill(-1), bd = new Float64Array(tiles).fill(Infinity)
  for (let i = 0; i < lab.length; i++) { const t = lab[i]!; let s2 = 0; for (let d = 0; d < 3; d++) s2 += (P[3 * i + d]! - c[3 * t + d]! / w[t]!) ** 2; if (s2 < bd[t]!) { bd[t] = s2; best[t] = i } }
  return best
}
function adjacency(faces: Int32Array, lab: Int32Array) {
  const s = new Set<string>()
  for (let q = 0; q < faces.length; q += 3) for (let k = 0; k < 3; k++) { const a = lab[faces[q + k]!]!, b = lab[faces[q + (k + 1) % 3]!]!; if (a !== b) s.add(a < b ? `${a} ${b}` : `${b} ${a}`) }
  return [...s].map((x) => x.split(' ').map(Number)).sort((u, v) => u[0]! - v[0]! || u[1]! - v[1]!).flat()
}

function emitLadder(L: any, P: Float64Array, faces: Int32Array, mass: Float64Array) {
  out += `levels ${L.levels.length} ${L.area_m2}\n`
  L.levels.forEach((l: any, k: number) => {
    emit(`level${k}`, 1, 11, [l.tiles, l.lambda_min, l.lambda_max, l.wavelength_m, l.diameter_m, l.diameter_in_wavelengths, l.tau, l.area_per_tile.cv, l.area_per_tile.min, l.area_per_tile.max, l.finest_tiles])
    emit(`labels${k}`, 1, L.n_vertices, L.labels[k]); emit(`areas${k}`, 1, l.tiles, L.tileAreas[k]); emit(`counts${k}`, 1, l.tiles, L.tileVertices[k])
    emit(`centres${k}`, 1, l.tiles, centres(P, L.labels[k], l.tiles, mass))
    const adj = adjacency(faces, L.labels[k]); emit(`adjacency${k}`, adj.length / 2, 2, adj)
  })
}

function mesh(name: string, positions: Float64Array, faces: Int32Array, n: number, kitFrames: boolean) {
  const mesh = { positions, faces, n }
  const L = surfaceLadder(mesh)
  const mass = vertexAreas(mesh)
  out += `mesh ${name} ${n}\n`
  emit('positions', n, 3, positions); emit('faces', faces.length / 3, 3, faces)
  emit('mass', 1, n, mass)
  emitLadder(L, positions, faces, mass)
  /* a scalar map and its tile statistics at every level (nxr map_stats) */
  const x = Float64Array.from({ length: n }, (_, i) => Math.sin(3 * positions[3 * i]!) + 0.5 * positions[3 * i + 2]! + (i % 7 === 0 ? NaN : 0))
  emit('x', 1, n, x)
  L.levels.forEach((l: any, k: number) => {
    const lab = L.labels[k] as Int32Array, T = l.tiles
    const st = { w: new Float64Array(T), s1: new Float64Array(T), s2: new Float64Array(T), n: new Float64Array(T), min: new Float64Array(T).fill(Infinity), max: new Float64Array(T).fill(-Infinity) }
    for (let i = 0; i < n; i++) if (Number.isFinite(x[i]!)) { const t = lab[i]!, a = mass[i]!; st.w[t]! += a; st.s1[t]! += a * x[i]!; st.s2[t]! += a * x[i]! * x[i]!; st.n[t]! += 1; st.min[t] = Math.min(st.min[t]!, x[i]!); st.max[t] = Math.max(st.max[t]!, x[i]!) }
    for (const key of Object.keys(st)) emit(`M${k}_${key}`, 1, T, (st as any)[key])
  })
  if (!kitFrames) return
  /* the canonical-like frame: the signpost basis turned by alpha(v); a tangent field with normal part */
  const kit = new SurfaceKit({ positions, faces })
  const b = kit.clOf(kit.components[0]).basis
  const alpha = Float64Array.from({ length: n }, (_, i) => 0.4 + 1.1 * Math.sin(2 * positions[3 * i + 1]!) + 0.7 * positions[3 * i]!)
  const e1 = new Float64Array(3 * n), e2 = new Float64Array(3 * n), nn = new Float64Array(3 * n), v = new Float64Array(3 * n)
  for (let i = 0; i < n; i++) {
    const ca = Math.cos(alpha[i]!), sa = Math.sin(alpha[i]!)
    for (let d = 0; d < 3; d++) {
      const a = b.e1[3 * i + d]!, c = b.e2[3 * i + d]!
      e1[3 * i + d] = ca * a + sa * c; e2[3 * i + d] = -sa * a + ca * c; nn[3 * i + d] = b.n[3 * i + d]!
    }
    /* a smooth tangent field of the signpost basis (angle beta) + a normal part */
    const beta = 0.8 * Math.cos(3 * positions[3 * i + 2]!) + positions[3 * i]!, mg = 1 + 0.3 * positions[3 * i + 1]!
    for (let d = 0; d < 3; d++) v[3 * i + d] = mg * (Math.cos(beta) * b.e1[3 * i + d]! + Math.sin(beta) * b.e2[3 * i + d]!) + 0.2 * b.n[3 * i + d]!
  }
  emit('alpha', 1, n, alpha); emit('e1', n, 3, e1); emit('e2', n, 3, e2); emit('nrm', n, 3, nn); emit('v', n, 3, v)
  const c1 = new Float64Array(n), c2 = new Float64Array(n), cn = new Float64Array(n)
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) { c1[i]! += v[3 * i + d]! * e1[3 * i + d]!; c2[i]! += v[3 * i + d]! * e2[3 * i + d]!; cn[i]! += v[3 * i + d]! * nn[3 * i + d]! }
  const lv = Math.min(LC_LEVELS, L.levels.length - 1)
  out += `lc ${lv}\n`
  for (let k = 1; k <= lv; k++) {
    const tiles = 2 ** k, lab = L.labels[k] as Int32Array, cen = centres(positions, lab, tiles, mass)
    const rho = new Float64Array(n).fill(NaN), omega = new Float64Array(tiles * tiles).fill(NaN)
    const angs: Float64Array[] = []
    for (let t = 0; t < tiles; t++) {
      const c = cen[t]!, X = kit.transported(kit.components[0], c, alpha[c]!)
      const ang = Float64Array.from({ length: n }, (_, i) => Math.atan2(X[2 * i + 1]!, X[2 * i]!) - alpha[i]!)
      const a0 = ang[c]!
      for (let i = 0; i < n; i++) ang[i] = Math.atan2(Math.sin(ang[i]! - a0), Math.cos(ang[i]! - a0)) // wrapped to (-pi, pi]
      angs.push(ang)
      for (let i = 0; i < n; i++) if (lab[i] === t) rho[i] = ang[i]!
    }
    for (let a = 0; a < tiles; a++) for (let bb = 0; bb < tiles; bb++) omega[a * tiles + bb] = angs[a]![cen[bb]!]!
    /* the sums: nxr vector_stats (closed form) and lc_resultant (Levi-Civita) */
    const S = { w: new Float64Array(tiles), c1: new Float64Array(tiles), c2: new Float64Array(tiles), cn: new Float64Array(tiles), c11: new Float64Array(tiles), c22: new Float64Array(tiles), c12: new Float64Array(tiles), mag: new Float64Array(tiles) }
    const re = new Float64Array(tiles), im = new Float64Array(tiles)
    for (let i = 0; i < n; i++) {
      const t = lab[i]!, a = mass[i]!
      S.w[t]! += a; S.c1[t]! += a * c1[i]!; S.c2[t]! += a * c2[i]!; S.cn[t]! += a * cn[i]!; S.c11[t]! += a * c1[i]! * c1[i]!; S.c22[t]! += a * c2[i]! * c2[i]!; S.c12[t]! += a * c1[i]! * c2[i]!; S.mag[t]! += a * Math.hypot(c1[i]!, c2[i]!)
      const ph = -rho[i]!; re[t]! += a * (c1[i]! * Math.cos(ph) - c2[i]! * Math.sin(ph)); im[t]! += a * (c1[i]! * Math.sin(ph) + c2[i]! * Math.cos(ph))
    }
    out += `level ${k}\n`
    emit('rho', 1, n, rho); emit('omega', tiles, tiles, omega)
    for (const key of Object.keys(S)) emit(`S_${key}`, 1, tiles, (S as any)[key])
    emit('lc_re', 1, tiles, re); emit('lc_im', 1, tiles, im)
  }
}

mesh('sphere', ...(() => { const S = sphere(2); return [S.positions, S.faces, S.n, true] as const })())
{
  const J = sphere(2), JP = Float64Array.from(J.positions)
  for (let v = 0; v < J.n; v++) { JP[3 * v] = JP[3 * v]! * 1.6 * (1 + 0.15 * rnd()); JP[3 * v + 1] = JP[3 * v + 1]! * (1 + 0.15 * rnd()); JP[3 * v + 2] = JP[3 * v + 2]! * 0.7 * (1 + 0.15 * rnd()) }
  mesh('irregular', JP, J.faces, J.n, true)
}
/* two pieces (a hemisphere pair): the SUBJECT tree's codes */
{
  const A = sphere(1), B = sphere(2), n = A.n + B.n
  const P = new Float64Array(3 * n); P.set(A.positions); for (let i = 0; i < B.n; i++) { P[3 * (A.n + i)] = B.positions[3 * i]! + 3; P[3 * (A.n + i) + 1] = B.positions[3 * i + 1]!; P[3 * (A.n + i) + 2] = B.positions[3 * i + 2]! }
  const F = Int32Array.from([...A.faces, ...Array.from(B.faces, (q) => q + A.n)])
  const Bt = P.slice(3 * A.n)     // the second piece AS STORED (translated): the ladder is run on the coordinates it will be given
  const la = surfaceLadder({ positions: A.positions, faces: A.faces, n: A.n }), lb = surfaceLadder({ positions: Bt, faces: B.faces, n: B.n })
  const finest = Math.min(la.levels.length, lb.levels.length) - 1
  out += `tree two ${n} ${A.n} ${finest}\n`
  emit('positions', n, 3, P); emit('faces', F.length / 3, 3, F)
  for (let d = 0; d <= finest; d++) emit(`codes${d}`, 1, n, [...Array.from(la.labels[d]!), ...Array.from(lb.labels[d]!, (t: number) => 2 ** d + t)])
}
writeFileSync(process.argv[2]!, out)
