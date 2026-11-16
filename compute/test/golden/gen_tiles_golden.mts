// Writes tiles_golden.txt: the Studio's reduceTiles (backend/src/db/measurements.ts), the reference of nxr::partition::reduceTiles: a segmented reduce over
// contiguous time windows, pooled over the rows. Cases: 3 rows x 1000 samples in windows of 64 (16 tiles, the last partial); one row (a value per row per
// window, as an array-backed measurement); and 4 rows x 4096 samples in windows of 256. Per case the values in and min, max, sum, sumSq, count out.
// Run:  NXR_CORTICAL_FLOW=<path to nxr-cortical-flow> $NXR_CORTICAL_FLOW/node_modules/.bin/tsx compute/test/golden/gen_tiles_golden.mts compute/test/golden/tiles_golden.txt
import { writeFileSync } from 'node:fs'
const root = process.env.NXR_CORTICAL_FLOW
if (!root) throw new Error('set NXR_CORTICAL_FLOW to the Studio monorepo checkout')
const { reduceTiles } = await import(`${root}/backend/src/db/measurements.ts`)
let seed = 7; const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5 }
const f = (a: ArrayLike<number>) => Array.from(a, (x) => String(x)).join(' ')
let out = ''
for (const [rows, samples, width, tiles] of [[3, 1000, 64, 16], [1, 1000, 64, 16], [4, 4096, 256, 16]]) {
  const x = new Float32Array(rows * samples)
  for (let i = 0; i < x.length; i++) x[i] = rnd() * 4
  const r = reduceTiles(x, rows, samples, width, tiles)
  out += `case ${rows} ${samples} ${width} ${tiles}\n${f(x)}\n${f(r.min)}\n${f(r.max)}\n${f(r.sum)}\n${f(r.sumSq)}\n${f(r.count)}\n`
}
writeFileSync(process.argv[2], out)
console.log(`wrote ${process.argv[2]}`)
