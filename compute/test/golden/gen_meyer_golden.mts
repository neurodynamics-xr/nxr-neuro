// Writes meyer_golden.txt: the Studio's own dyadic frame (wavelets/dyadicFrame.ts) evaluated on a grid, for test_filterbank.cpp.
// Run:  NXR_CORTICAL_FLOW=<path to nxr-cortical-flow> $NXR_CORTICAL_FLOW/node_modules/.bin/tsx compute/test/golden/gen_meyer_golden.mts compute/test/golden/meyer_golden.txt
import { writeFileSync } from 'node:fs'
const root = process.env.NXR_CORTICAL_FLOW
if (!root) throw new Error('set NXR_CORTICAL_FLOW to the Studio monorepo checkout')
const { weylFrame, dyadicFrame, rankFrame, rankCoordinates, frameGains, meyerBand } =
  await import(`${root}/studio/src/engine/toolbox/wavelets/dyadicFrame.ts`)

const f = (a: ArrayLike<number>) => Array.from(a, (x) => x.toPrecision(17)).join(' ')
const grid = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, i) => lo * (hi / lo) ** (i / (n - 1)))
let out = ''
const emit = (name: string, frame: any, xs: number[]) => {
  const g = frameGains(frame, xs)
  out += `frame ${name} ${frame.members.length} ${xs.length} ${frame.halfWidth.toPrecision(17)} ${frame.edges.length}\nedges ${f(frame.edges)}\nx ${f(xs)}\n`
  for (const row of g) out += `g ${f(row)}\n`
}
const xs = [0, ...grid(0.01, 500, 120), 0.0, -1]
emit('weyl', weylFrame(1.8, 6), xs.concat(Array.from({ length: 6 }, (_, j) => 4 * Math.PI * 2 ** j / 1.8)))     // area 1.8, 6 levels; the edges themselves too
emit('dyadic-quarter', dyadicFrame([0.5, 2, 8, 32], { scaling: 0, wavelet: (k: number) => k, top: 4 }, 0.25), xs)
emit('rank', rankFrame(4), grid(0.5, 40, 60))
const lam = [3, 1, 2, 2, 5, 1, 1, 9, 0.5, 4, 4, 4]
out += `rankcoords ${lam.length} ${f(lam)}\nmult1 ${f(rankCoordinates(lam, 1))}\nmult4 ${f(rankCoordinates(lam, 4))}\n`
out += `band ${f([0, 0.3, 0.7, 1, 1.4, 2, 3, 100].map((x) => meyerBand(x, 0.5, 2, 1 / 3)))}\n`
writeFileSync(process.argv[2], out)
