// IMPORTS through wasmClient against the Studio's own service: the same Brainstorm source imported into two copies of one datastore — by the Studio's
// `importSubjects` running the NATIVE `nxr-neuro` as its converter, and by `wasmClient.imports.post`, which runs the same command line compiled into the
// module — then every table and file compared. Both sides deterministic (NXR_IO_DETERMINISTIC: counter ids, a fixed clock), so the converter's rows match exactly.
//   npx tsx <this> <studio checkout> <wasm-io build folder> <datastore> <protocols> <native nxr-neuro>
import assert from 'node:assert/strict';
import { cpSync, readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import path from 'node:path';
import { keysOf, readKey } from './store_keys.mjs';   // the store's KEYS, at format 51 (files) and 52 (rows of the subject's small_files.sqlite)

const [, , STUDIO, PKG, STORE, PROTOCOLS, NATIVE] = process.argv;
const { createModelService } = await import(path.join(STUDIO, 'backend/src/service/model.ts'));
const { wasmClient } = await import(path.join(PKG, 'index.mjs'));
const A = `${STORE}-imp-studio`, B = `${STORE}-imp-wasm`;
cpSync(STORE, A, { recursive: true });
cpSync(STORE, B, { recursive: true });
const SEED = '1000';
// the converter inherits this process's environment (the service REPLACES a converter's own `env` with its lock handshake while it holds the dataset)
process.env.NXR_IO_DETERMINISTIC = SEED;
const studio = createModelService({ converter: { cmd: NATIVE } });
await studio.open(A);
const wasm = await wasmClient(B, { program: 'cortical-flow' });
let checks = 0;

const TIME = /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$/;
const deep = (v: unknown): unknown => typeof v === 'string' && TIME.test(v) ? '<T>' : Array.isArray(v) ? v.map(deep)
  : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, deep(x)])) : v;
const rows = (root: string, sql: string) => {
  const cat = JSON.parse(readFileSync(path.join(root, 'catalog.json'), 'utf8'));
  const db = new DatabaseSync(path.join(root, cat.datasets[0].database), { readOnly: true });
  try { return JSON.parse(JSON.stringify(db.prepare(sql).all())); } finally { db.close(); }
};
function same(what: string) {
  const tables = (rows(A, "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name") as { name: string }[]).map((t) => t.name);
  // the converter's ids are the deterministic counter on both sides; what the services mint (the band and measure dictionaries) is named
  const minted = (root: string) => new Map((rows(root, "SELECT id, name FROM dictionary WHERE scope = 'app'") as { id: string; name: string }[]).map((r) => [r.id, `<${r.name}>`]));
  const ia = minted(A), ib = minted(B);
  const norm = (r: unknown[], ids: Map<string, string>) => { let t = JSON.stringify(deep(r)); for (const [id, l] of ids) t = t.split(id).join(l); return (JSON.parse(t) as unknown[]).map((x) => JSON.stringify(x)).sort(); };
  for (const t of tables) {
    const q = t === 'datastore' ? 'SELECT id, driver FROM datastore' : `SELECT * FROM "${t}"`;
    assert.deepEqual(norm(rows(B, q), ib), norm(rows(A, q), ia), `${what}: table ${t}`);
  }
  const files = (root: string) => keysOf(root).filter((f) => !/dataset\.sqlite|catalog\.json|\.lock$/.test(f));
  assert.deepEqual(files(B), files(A), `${what}: files`);
  // every zarr.json alike (its text, times masked): the converter lays the stores out on both sides
  for (const f of files(A).filter((x) => x.endsWith('zarr.json')))
    assert.deepEqual(deep(JSON.parse(readKey(B, f).toString('utf8'))), deep(JSON.parse(readKey(A, f).toString('utf8'))), `${what}: ${f}`);
  checks++;
}

// each import its own seed: the counter restarts at the seed every run, so two imports with one seed would mint the same ids
const seed = (n: string) => { process.env.NXR_IO_DETERMINISTIC = n; wasm.setEnv('NXR_IO_DETERMINISTIC', n); };
seed('1000');
const session = { dataset: 'd', source: { kind: 'brainstorm', root: path.join(PROTOCOLS, 'session'), subject: 'S01', condition: 'sub-S01_ses-01_task-rest_run-01_meg' } };
const e1 = await studio.importSubjects(session);
const w1 = await wasm.imports.post(session);
assert.deepEqual(deep(w1), deep(JSON.parse(JSON.stringify(e1)))); checks++;
same('a Brainstorm subject and its condition');
// again, replacing it (every session): the converter's `subject delete`, then the conversion
const again = { ...session, replace: true, replaceAllSessions: true };
await assert.rejects(Promise.resolve().then(() => studio.importSubjects({ ...again, replaceAllSessions: false })), /replaceAllSessions/);
await assert.rejects(wasm.imports.post({ ...again, replaceAllSessions: false }), /replaceAllSessions/); checks++;
seed('2000');
const e2 = await studio.importSubjects(again);
const w2 = await wasm.imports.post(again);
assert.deepEqual(deep(w2), deep(JSON.parse(JSON.stringify(e2)))); checks++;
same('replaced');
// the template (the protocol's default anatomy)
const tpl = { dataset: 'd', source: { kind: 'brainstorm-template', root: path.join(PROTOCOLS, 'full') } };
seed('3000');
const e3 = await studio.importSubjects(tpl);
const w3 = await wasm.imports.post(tpl);
assert.deepEqual(deep(w3), deep(JSON.parse(JSON.stringify(e3)))); checks++;
same('a template');
await assert.rejects(Promise.resolve().then(() => studio.importSubjects(tpl)), /already a subject/);
await assert.rejects(wasm.imports.post(tpl), /already a subject/); checks++;
// a v7.3 .mat (an HDF5 file: here its 128-byte MATLAB header, enough to be told apart) is refused in WebAssembly by name — matio is built without HDF5
const { writeFileSync } = await import('node:fs');
const v73 = path.join(PROTOCOLS, 'tess_v73.mat');
const head = Buffer.alloc(128, 0x20); head.write('MATLAB 7.3 MAT-file, Platform: x, Created on: x HDF5 schema 1.00 .'); head.writeUInt16LE(0x0200, 124); head.write('IM', 126);
writeFileSync(v73, Buffer.concat([head, Buffer.alloc(512)]));
const r = await wasm.cli(['import', 'surface', '--dataset', path.join(B, 'd'), '--subject', 'S01', '--file', v73]);
assert.equal(r.code, 1, JSON.stringify(r));
assert.match(JSON.stringify(r.lines), /MATLAB v7\.3 \(HDF5\) file, which nxr-neuro's WebAssembly build does not read/); checks++;
wasm.close();
console.log(`wasm-io imports vs the Studio's service: ${checks} checks passed`);
