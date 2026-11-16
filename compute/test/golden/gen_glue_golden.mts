// Writes glue_golden.txt: the Studio's glueMesh and sourcesOf (studio/src/studio/glue.ts), the reference of nxr::glue, on synthetic two-piece closed
// surfaces: two bumpy ellipsoids side by side, each with a MEDIAL WALL (the cap facing the other) and a stray wall vertex elsewhere (the cut keeps the
// largest patch). Case 0: equal meshes; case 1: unequal (another subdivision level and size), so the rims differ. Per case: the input (positions, faces,
// wall) and the output (positions, faces, keep, the bridge's vertices · rings · area · longest edge, the sources' idx and w).
// Run:  NXR_CORTICAL_FLOW=<path to nxr-cortical-flow> $NXR_CORTICAL_FLOW/node_modules/.bin/tsx compute/test/golden/gen_glue_golden.mts compute/test/golden/glue_golden.txt
import { writeFileSync } from 'node:fs'
const root = process.env.NXR_CORTICAL_FLOW
if (!root) throw new Error('set NXR_CORTICAL_FLOW to the Studio monorepo checkout')
const glue = await import(`${root}/studio/src/studio/glue.ts`)

function sphere(sub: number) {
  const t = (1 + Math.sqrt(5)) / 2
  const verts: number[][] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
  let tris = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
  for (let s = 0; s < sub; s++) {
    const mid = new Map<string, number>()
    const mp = (a: number, b: number) => { const k = a < b ? `${a}_${b}` : `${b}_${a}`; const h = mid.get(k); if (h !== undefined) return h; verts.push([0, 1, 2].map((c) => (verts[a]![c]! + verts[b]![c]!) / 2)); mid.set(k, verts.length - 1); return verts.length - 1 }
    tris = tris.flatMap(([a, b, c]) => { const ab = mp(a!, b!), bc = mp(b!, c!), ca = mp(c!, a!); return [[a!, ab, ca], [b!, bc, ab], [c!, ca, bc], [ab, bc, ca]] })
  }
  return { verts: verts.map((p) => { const l = Math.hypot(p[0]!, p[1]!, p[2]!); return p.map((x) => x / l) }), tris }
}
/** a bumpy ellipsoid at `off` (x), radii D × `size`; the wall: the cap facing x = 0 (|x − off| side nearest the gap) */
function hemi(sub: number, off: number, size: number, sign: number) {
  const s = sphere(sub)
  const P = s.verts.map(([x, y, z]) => { const b = 1 + 0.08 * Math.sin(3 * x! + 2 * y!) * Math.cos(2 * z!); return [off + size * 0.6 * x! * b, size * 1.0 * y! * b, size * 0.8 * z! * b] })
  const wall = s.verts.map(([x], i) => (sign * x! > 0.72 ? 1 : 0))
  wall[Math.floor(s.verts.length / 3)] = 1                                             // a stray wall vertex: not part of the cut patch
  return { P, F: s.tris, wall }
}
const f = (a: ArrayLike<number>) => Array.from(a, (x) => (Number.isFinite(x) ? Number(x).toPrecision(17) : 'nan')).join(' ')
let out = ''
const emit = (name: string, rows: number, cols: number, a: ArrayLike<number>) => { out += `${name} ${rows} ${cols}\n${f(a)}\n` }
function kase(a: ReturnType<typeof hemi>, b: ReturnType<typeof hemi>) {
  const P = new Float64Array([...a.P.flat(), ...b.P.flat()]), n = a.P.length
  const F = Int32Array.from([...a.F.flat(), ...b.F.flat().map((v) => v + n)]), wall = Uint8Array.from([...a.wall, ...b.wall])
  const g = glue.glueMesh(P, F, wall)
  /* the sources (sourcesOf is the module's own): through glueSource, a Field carried onto the glued surface — a bridge vertex's value is the
     inverse-distance blend of its 3 nearest cut wall vertices; f(v) = v + 1 so the blend is read back exactly enough (float32) */
  const values = Float32Array.from({ length: n + b.P.length }, (_, v) => v + 1)
  const port = glue.glueSource({ manifolds: [{ kind: 'surface', id: 's', positions: P, faces: F }], labellings: [{ of: 's', name: 'Desikan-Killiany', codes: Int32Array.from(wall), entries: [{ code: 0, name: 'cortex' }, { code: 1, name: 'unassigned' }] }],
    fields: [{ id: 'f', on: 's', type: 'real', values }] } as any)
  const carried = port.fields[0].values as Float32Array
  emit('positions', P.length / 3, 3, P); emit('faces', F.length / 3, 3, F); emit('wall', 1, wall.length, wall)
  emit('out_positions', g.positions.length / 3, 3, g.positions); emit('out_faces', g.faces.length / 3, 3, g.faces); emit('out_keep', 1, g.keep.length, g.keep)
  emit('bridge', 1, 4, [g.bridge.vertices, g.bridge.rings, g.bridge.area, g.bridge.longestEdge])
  emit('carried', 1, carried.length, carried)
}
kase(hemi(3, -0.75, 1.0, 1), hemi(3, 0.75, 1.0, -1))
kase(hemi(3, -0.8, 1.0, 1), hemi(2, 0.7, 0.85, -1))
writeFileSync(process.argv[2]!, `cases 2\n` + out)
console.log(`wrote ${process.argv[2]}`)
