// Writes partition_golden.txt: the Studio's measureOverPartition (backend/src/db/measurements.ts), the reference of nxr::partition, on synthetic
// partitions: per case the codes, values and weights in, and per member every measure it writes (NaN where it writes none), with its row weight.
// Cases: unit weights (Brainstorm's plain mean), positive weights (an area), and an EMPTY member (a code with no element).
// Run:  NXR_CORTICAL_FLOW=<path to nxr-cortical-flow> $NXR_CORTICAL_FLOW/node_modules/.bin/tsx compute/test/golden/gen_partition_golden.mts compute/test/golden/partition_golden.txt
import { writeFileSync } from 'node:fs'
const root = process.env.NXR_CORTICAL_FLOW
if (!root) throw new Error('set NXR_CORTICAL_FLOW to the Studio monorepo checkout')
const { measureOverPartition } = await import(`${root}/backend/src/db/measurements.ts`)

const MEASURES = ['mean', 'mean_square', 'max', 'min', 'abs_max', 'sum', 'area', 'n_elements']
const f = (a: ArrayLike<number>) => Array.from(a, (x) => (Number.isFinite(x) ? Number(x).toPrecision(17) : 'nan')).join(' ')
let out = ''
const emit = (name: string, rows: number, cols: number, a: ArrayLike<number>) => { out += `${name} ${rows} ${cols}\n${f(a)}\n` }
let seed = 11; const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 }

function kase(n: number, members: number, weighted: boolean, empty: number | null) {
  const codes = new Int32Array(n), values = new Float64Array(n), weights = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let c = Math.floor(rnd() * members); if (c === empty) c = (c + 1) % members
    codes[i] = c; values[i] = 3 * (rnd() - 0.4) * (1 + c); weights[i] = weighted ? 1e-6 * (0.2 + rnd()) : 1
  }
  const rows = measureOverPartition({ selectionId: 's', ofFieldId: 'f', labels: codes, nMembers: members, values, weights: weighted ? weights : undefined, measures: MEASURES })
  const table = new Float64Array(members * MEASURES.length).fill(NaN), w = new Float64Array(members).fill(NaN)
  for (const r of rows) { table[r.code! * MEASURES.length + MEASURES.indexOf(r.measure)] = r.value!; w[r.code!] = r.weight! }
  out += `case ${n} ${members} ${weighted ? 1 : 0}\n`
  emit('codes', 1, n, codes); emit('values', 1, n, values); emit('weights', 1, n, weights)
  emit('measures', members, MEASURES.length, table); emit('row_weight', 1, members, w)
}
kase(200, 7, false, null)
kase(500, 12, true, null)
kase(300, 9, true, 4)
out = `cases 3\n` + out
writeFileSync(process.argv[2]!, out)
console.log(`wrote ${process.argv[2]}`)
