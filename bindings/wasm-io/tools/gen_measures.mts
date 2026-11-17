// io/schema/measures.json from the Cortical Flow Studio's measure registry (backend/src/db/measurements.ts: MEASURES), the reference. The derived measures'
// formulas are code (nxr/io/measures.h, by name); everything else is this data. The Studio's `reduction` is written as `merge` (the one rule key io's SQL and compute's roll-up share); `form` is the form of the field a measure is taken of. Regenerate when the Studio's registry changes; the writes test checks it.
//   (cd <studio> && npx tsx <this> <studio> <nxr-neuro>/io/schema/measures.json)
import { writeFileSync } from 'node:fs';
import path from 'node:path';
const [, , STUDIO, OUT] = process.argv;
const { MEASURES, MEASURE_DICTIONARY, measureDictionaryRows } = await import(path.join(STUDIO, 'backend/src/db/measurements.ts'));
const d = measureDictionaryRows({ id: 'ID' });
const out = {
  source: 'nxr-cortical-flow backend/src/db/measurements.ts (MEASURES, measureDictionaryRows)',
  dictionary: { name: MEASURE_DICTIONARY, scope: d.dictionary.scope, description: d.dictionary.description },
  measures: MEASURES.map((m: { code: number; name: string; reduction: string | null; form?: string; unit: string | null; weight?: string; span?: string; derived?: { from: string[] }; description: string }) => ({
    code: m.code, name: m.name, merge: m.reduction ?? 'derived', form: m.form ?? 'none', unit: m.unit, weight: m.weight ?? null, span: m.span ?? null, derived_from: m.derived ? m.derived.from : null, description: m.description,
  })),
};
writeFileSync(OUT, JSON.stringify(out, null, 1) + '\n');
console.log(`${out.measures.length} measures → ${OUT}`);
