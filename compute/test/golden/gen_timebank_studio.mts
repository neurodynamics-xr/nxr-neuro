// Writes timebank_studio_golden.txt: the Studio's own time-axis gains — windowFrame (dyadicFrame.ts), the wavelet chain's bankGain / morletHat / meyerHat /
// waveletFreqs (signal/chain.ts) — evaluated on a grid, for test_timebank.cpp.
// Run:  NXR_CORTICAL_FLOW=<path to nxr-cortical-flow> $NXR_CORTICAL_FLOW/node_modules/.bin/tsx compute/test/golden/gen_timebank_studio.mts compute/test/golden/timebank_studio_golden.txt
import { writeFileSync } from 'node:fs'
const root = process.env.NXR_CORTICAL_FLOW
if (!root) throw new Error('set NXR_CORTICAL_FLOW to the Studio monorepo checkout')
const dy = await import(`${root}/studio/src/engine/toolbox/wavelets/dyadicFrame.ts`)
const ch = await import(`${root}/studio/src/engine/toolbox/signal/chain.ts`)

const f = (a: ArrayLike<number>) => Array.from(a, (x) => x.toPrecision(16)).join(' ')
const fs = 100, n = 192
const grid = Array.from({ length: n / 2 + 1 }, (_, k) => (k * fs) / n)
let out = ''
const wf = dy.windowFrame(fs, n)
out += `window_edges 1 ${wf.edges.length}\n${f(wf.edges)}\n`
const g = dy.frameGains(wf, grid)
out += `window_gains ${grid.length} ${g.length}\n`
for (let k = 0; k < grid.length; k++) out += f(g.map((r: ArrayLike<number>) => r[k]!)) + ' '
out += '\n'
const fc = ch.waveletFreqs(fs, { f0: 2, f1: 40, voices: 4 })
out += `fc 1 ${fc.length}\n${f(fc)}\n`
const mor = (fcm: number) => grid.map((x) => ch.bankGain({ mother: 'morlet', voices: 4 }, x, fcm))
out += `bank_morlet ${grid.length} ${fc.length}\n`
for (let k = 0; k < grid.length; k++) out += f(Array.from(fc, (c: number) => mor(c)[k]!)) + ' '
out += '\n'
const mey = (fcm: number) => grid.map((x) => ch.bankGain({ mother: 'meyer', voices: 4 }, x, fcm))
out += `bank_meyer ${grid.length} ${fc.length}\n`
for (let k = 0; k < grid.length; k++) out += f(Array.from(fc, (c: number) => mey(c)[k]!)) + ' '
out += '\n'
// is the Studio's voices-an-octave Meyer mother bank tight? sum of (g/2)^2 over a log grid, interior of the bank
const fine = Array.from({ length: 400 }, (_, i) => 4 * 2 ** (i / 400 * 3))
const S = fine.map((x) => Array.from(fc, (c: number) => (ch.bankGain({ mother: 'meyer', voices: 4 }, x, c) / 2) ** 2).reduce((a: number, b: number) => a + b, 0))
out += `meyer_mother_sum 1 2\n${Math.min(...S).toPrecision(16)} ${Math.max(...S).toPrecision(16)}\n`
writeFileSync(process.argv[2], out)
