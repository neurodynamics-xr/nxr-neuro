// Writes constant_q_golden.json: the Studio's constant-Q tiling (backend/src/db/constant-q.ts), the reference of nxr::io::constantq, on synthetic time lines.
// Per case: the line and params in; out the tiling (bands and levels), the level dictionary's entries (constantQDictionaryRows: name, measurement,
// attributes_json), the ladder's row (constantQRows: label, description, params_json) and its members' (name, label, n_members, params_json).
// Run:  NXR_CORTICAL_FLOW=<path to nxr-cortical-flow> $NXR_CORTICAL_FLOW/node_modules/.bin/tsx io/test/golden/gen_constant_q_golden.mts io/test/golden/constant_q_golden.json
import { writeFileSync } from 'node:fs'
const root = process.env.NXR_CORTICAL_FLOW
if (!root) throw new Error('set NXR_CORTICAL_FLOW to the Studio monorepo checkout')
const cq = await import(`${root}/backend/src/db/constant-q.ts`)
const cases = [
  { name: 'rest1200', line: { sfreq: 1200, samples: 360000, origin: 0 }, params: {} },
  { name: 'meg2400', line: { sfreq: 2400, samples: 1440000, origin: 0 }, params: {} },
  { name: 'eeg250', line: { sfreq: 250, samples: 75000, origin: 1.5 }, params: {} },
  { name: 'cycles1200', line: { sfreq: 1200, samples: 360000, origin: 0 }, params: { method: 'cycles' } },
  { name: 'voices3', line: { sfreq: 1000, samples: 50000, origin: 0 }, params: { voices: 3, fMin: 2 } },
]
let n = 0
const newId = () => `id${n++}`
const out = cases.map((c) => {
  const tiling = cq.constantQTiling(c.line, c.params)
  const dict = cq.constantQDictionaryRows('subj', tiling, { name: 'line_constantQ', newId })
  const rows = cq.constantQRows('subj', 'line', tiling, { name: 'line_constantQ', newId })
  return {
    name: c.name, line: c.line, params: c.params, tiling,
    entries: dict.entries.map((e: any) => ({ code: e.code, name: e.name, measurement: e.measurement, attributes_json: e.attributes_json })),
    set: { label: rows.set.label, description: rows.set.description, params_json: rows.set.params_json, n_members: rows.set.n_members, n_elements: rows.set.n_elements },
    members: rows.members.map((m: any) => ({ name: m.name, label: m.label, code: m.code, n_members: m.n_members, params_json: m.params_json })),
    tileDescriptions: tiling.levels.map((l: any) => cq.levelTileDictionaryRows('subj', l, { name: 'x', newId }).description),
  }
})
writeFileSync(process.argv[2], JSON.stringify(out, null, 1))
console.log(`wrote ${process.argv[2]}: ${out.length} cases, levels ${out.map((o) => o.tiling.levels.length).join(' ')}`)
