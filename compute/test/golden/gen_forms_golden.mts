// Writes forms_golden.txt: the Studio's field forms (studio/src/engine/toolbox/differential/fieldForms.ts `formValues`, on faceGradient.ts
// `assembleFaceGradient`), the reference of nxr::forms, on a synthetic closed surface: a bumpy icosphere (level 3, float32 positions, CCW outward) with
// random fields of each type, two frames each: ℝ (1 a vertex), ℂ (3 a vertex, an embedded vector), ℍ ([w x y z]). Per type and form: the form's values.
// Also the mesh's mean curvature (`meanCurvatureCpu`), its vertex normals and signed volume. Numbers are written exactly (a float32 as its shortest
// double repr; read back by strtod and rounded to float, it is the same bits).
// Run:  NXR_CORTICAL_FLOW=<path to nxr-cortical-flow> $NXR_CORTICAL_FLOW/node_modules/.bin/tsx compute/test/golden/gen_forms_golden.mts compute/test/golden/forms_golden.txt
import { writeFileSync } from 'node:fs'
const root = process.env.NXR_CORTICAL_FLOW
if (!root) throw new Error('set NXR_CORTICAL_FLOW to the Studio monorepo checkout')
const fg = await import(`${root}/studio/src/engine/toolbox/differential/faceGradient.ts`)
const ff = await import(`${root}/studio/src/engine/toolbox/differential/fieldForms.ts`)

// an icosphere, subdivided `level` times, then bumped radially (r = 1 + 0.15 sin(3x) cos(2z)), so its curvature varies
function icosphere(level: number) {
  const t = (1 + Math.sqrt(5)) / 2
  let V: number[][] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
  let F: number[][] = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
  const norm = (p: number[]) => { const l = Math.hypot(p[0], p[1], p[2]); return [p[0] / l, p[1] / l, p[2] / l] }
  V = V.map(norm)
  for (let s = 0; s < level; s++) {
    const mid = new Map<string, number>(), nF: number[][] = []
    const m = (a: number, b: number) => { const k = a < b ? `${a},${b}` : `${b},${a}`; let i = mid.get(k); if (i === undefined) { i = V.length; V.push(norm([(V[a][0] + V[b][0]) / 2, (V[a][1] + V[b][1]) / 2, (V[a][2] + V[b][2]) / 2])); mid.set(k, i) } return i }
    for (const [a, b, c] of F) { const ab = m(a, b), bc = m(b, c), ca = m(c, a); nF.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]) }
    F = nF
  }
  const P = new Float32Array(V.length * 3)
  V.forEach((p, i) => { const r = 1 + 0.15 * Math.sin(3 * p[0]) * Math.cos(2 * p[2]); P[3 * i] = p[0] * r; P[3 * i + 1] = p[1] * r; P[3 * i + 2] = p[2] * r })
  return { P, F: Int32Array.from(F.flat()), nV: V.length, nF: F.length }
}

// a seeded generator (the golden must be the same on every run)
let seed = 12345
const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296 - 0.5 }

const { P, F, nV, nF } = icosphere(3)
const op = fg.assembleFaceGradient(P, F, nV)
const H = fg.meanCurvatureCpu(op)
const out: string[] = []
const line = (tag: string, a: ArrayLike<number>) => out.push(`${tag} ${a.length} ${Array.from(a, (x) => String(x)).join(' ')}`)
out.push(`mesh ${nV} ${nF}`)
line('positions', P)
line('faces', F)
line('vertexNormals', op.vertexNormals)
line('meanCurvature', H)
out.push(`signedVolume ${op.signedVolume}`)
const frames = 2
for (const t of ['real', 'complex', 'quaternion'] as const) {
  const W = ff.widthOf(t, nV), values = new Float32Array(frames * W)
  for (let i = 0; i < values.length; i++) values[i] = rnd()
  line(`field ${t}`, values)
  for (const info of ff.FORMS[t]) line(`form ${t} ${info.id}`, ff.formValues(t, info.id, values, nV, op))
}
writeFileSync(process.argv[2], out.join('\n') + '\n')
console.log(`wrote ${process.argv[2]}: ${nV} vertices, ${nF} faces, ${ff.FORMS.real.length + ff.FORMS.complex.length + ff.FORMS.quaternion.length} forms`)
