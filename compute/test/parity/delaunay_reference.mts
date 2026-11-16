// The REFERENCE side of the Delaunay parity test: the Cortical Flow Studio's own intrinsicDelaunay.ts (the construction nxr-compute's delaunay.h is a
// line port of) on one mesh, printed as JSON. Run with the monorepo's tsx:
//
//   export NXR_CORTICAL_FLOW=<path to the nxr-cortical-flow checkout>
//   $NXR_CORTICAL_FLOW/node_modules/.bin/tsx delaunay_reference.mts <mesh.bin> <mollify> <flipEps> > reference.json
//
// mesh.bin: int32 nV, int32 nF, float64 positions [3 nV], int32 faces [3 nF] (little-endian) — the format delaunay_dump reads.
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const MONO = process.env.NXR_STUDIO ?? process.env.NXR_CORTICAL_FLOW ?? (() => { throw new Error('set NXR_CORTICAL_FLOW to a checkout of nxr-cortical-flow') })()
const D = await import(resolve(MONO, 'studio/src/engine/toolbox/differential/intrinsicDelaunay.ts'))
const [file, mollify, flipEps] = process.argv.slice(2)
const buf = readFileSync(file!)
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
const nV = dv.getInt32(0, true), nF = dv.getInt32(4, true)
const P = new Float64Array(3 * nV), F = new Int32Array(3 * nF)
for (let i = 0; i < 3 * nV; i++) P[i] = dv.getFloat64(8 + 8 * i, true)
for (let i = 0; i < 3 * nF; i++) F[i] = dv.getInt32(8 + 24 * nV + 4 * i, true)
const m = D.intrinsicMesh(P, F)
D.mollify(m, { mollify: Number(mollify) })
D.addSignposts(m)
D.flipToDelaunay(m, { flipEps: Number(flipEps) })
const o = m.options
process.stdout.write(JSON.stringify({
  F: Array.from(m.F), twin: Array.from(m.twin), length: Array.from(m.length), mass: Array.from(m.mass),
  angle: Array.from(m.sign.angle), angleSum: Array.from(m.sign.angleSum), ref: Array.from(m.sign.ref),
  record: { mollify: o.mollify, epsilon: o.epsilon, flipEps: o.flipEps, flips: o.flips }, nonDelaunay: D.nonDelaunayEdges(m, Number(flipEps)),
}) + '\n')
