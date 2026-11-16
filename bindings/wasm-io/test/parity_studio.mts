// wasmClient against the Cortical Flow Studio's own code, on a synthetic datastore (make_store.py). Run from the Studio's folder (its node_modules):
//   npx tsx <this> <studio checkout> <wasm-io build folder> <datastore>
// Checks: every subject's bundle equals the Studio's `bundleOf`; views and rows equal the service's `list` / `read` queries; every array a bundle names
// reads equal to the Studio's `read` (as stored, as float64, a slice); an array created and put through wasmClient reads back through the Studio.
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const [, , STUDIO, PKG, STORE] = process.argv;
const { bundleOf } = await import(path.join(STUDIO, 'backend/src/db/bundle.ts'));
const { openLocal } = await import(path.join(STUDIO, 'backend/src/store-node.ts'));
const { read } = await import(path.join(STUDIO, 'backend/src/read.ts'));
const { wasmClient, NxrIoError } = await import(path.join(PKG, 'index.mjs'));

const plain = (x: unknown) => JSON.parse(JSON.stringify(x));
const c = await wasmClient(STORE);
const catalog = JSON.parse(readFileSync(path.join(STORE, 'catalog.json'), 'utf8'));
let checks = 0;
const ok = () => { checks++; };

// status
const st = await c.status();
assert.deepEqual(plain(st.catalog), catalog); ok();

for (const d of catalog.datasets) {
  const db = new DatabaseSync(path.join(STORE, d.database), { readOnly: true });
  const one = (sql: string, ...a: unknown[]) => db.prepare(sql).get(...(a as never[])) as Record<string, unknown> | undefined;
  const src = {
    db,
    subjects: { read: (id: string) => one('SELECT * FROM subject WHERE id = ?', id) },
    datasets: { read: (id: string) => one('SELECT * FROM dataset WHERE id = ?', id) },
    datastores: { read: (id: string) => one('SELECT * FROM datastore WHERE id = ?', id) },
  };
  // views and tables: the service's `list` (one dataset here: its rows; a dataset row carries `writer_lock`, null with no writer)
  for (const v of ['manifold_view', 'recording_view', 'operator_view', 'selection_view', 'field_view', 'acquisition_view', 'subject_status', 'atlas_view', 'dashboard_counts',
                   'datastore', 'dataset', 'subject', 'acquisition', 'manifold', 'selection', 'field', 'operator', 'dictionary', 'coordinate_system', 'job']) {
    let want = plain(db.prepare(`SELECT * FROM ${v} LIMIT 5000`).all());
    if (v === 'dataset') want = want.map((r: object) => ({ ...r, writer_lock: null }));
    assert.deepEqual(plain(await c.views.get(v)), want, `view ${v}`); ok();
  }
  const sub0 = (db.prepare('SELECT * FROM subject ORDER BY name').all() as { id: string; path: string; name: string }[]);
  assert.deepEqual(plain(await c.views.get('subject', { name: sub0[0].name }, 1)), plain([sub0[0]])); ok();
  await assert.rejects(c.views.get('sqlite_master'), (e: InstanceType<typeof NxrIoError>) => e.code === 'invalid_argument'); ok();
  await assert.rejects(c.views.get('subject', { 'name; DROP': 1 }), (e: InstanceType<typeof NxrIoError>) => e.code === 'invalid_argument'); ok();

  for (const s of sub0) {
    assert.deepEqual(plain(await c.rows.get('subject', s.id)), plain(s)); ok();
    const storePath = `${d.path}/${s.path}`;
    // the bundle
    const want = plain(bundleOf(src, s.id));
    const got = plain(await c.subjects.get(storePath));
    assert.deepEqual(Object.keys(got).sort(), Object.keys(want).sort(), `${s.name}: bundle keys`);
    for (const k of Object.keys(want)) { assert.deepEqual(got[k], want[k], `${s.name}: bundle.${k}`); ok(); }
    // the stored STATISTICS: sub-1 has two complete rows (the pending one is left out), in the order path, name
    if (s.name === 'sub-1') { assert.deepEqual(got.statistics.map((r: { name: string }) => r.name), ['csd_a', 'psd_b'], 'sub-1: bundle.statistics'); ok(); }
    // the arrays it names
    const store = await openLocal(path.join(STORE, storePath));
    const paths = new Set<string>();
    for (const g of want.geometries) if (g.subject_id === s.id && g.positions_path) paths.add(g.positions_path);
    for (const t of want.topology_cells ?? []) if (t.path && want.topologies.some((x: { id: string; subject_id: string }) => x.id === t.topology_id && x.subject_id === s.id)) paths.add(t.path);
    for (const f of want.fields) if (f.path) paths.add(f.path);
    for (const x of want.selections) if (x.array_path) paths.add(x.array_path);
    for (const p of paths) {
      const e = await read(store, p);
      const w = await c.read(storePath, p);
      assert.equal(w.constructor.name, e.constructor.name, `${p}: type`);
      assert.deepEqual(Array.from(w as ArrayLike<number | bigint>, Number), Array.from(e as ArrayLike<number | bigint>, Number), `${p}: values`); ok();
      assert.deepEqual(Array.from(await c.read(storePath, p, { as: 'float64' })), Array.from(await read(store, p, { as: 'float64' })), `${p}: as float64`); ok();
    }
    // a slice of a 2-D array (the positions)
    const pos = want.geometries.find((g: { subject_id: string }) => g.subject_id === s.id)?.positions_path;
    if (pos) {
      const sl: [number, number][] = [[1, 3], [0, 3]];
      assert.deepEqual(Array.from(await c.read(storePath, pos, { slice: sl })), Array.from(await read(store, pos, { slice: sl })), `${pos}: slice`); ok();
    }
  }

  // ARRAY WRITES through wasmClient, read back by the Studio
  const s = sub0[0], storePath = `${d.path}/${s.path}`;
  const data = Float32Array.from({ length: 12 }, (_, i) => i * 0.5 - 1);
  const made = await c.arrays.create(storePath, 'scratch/wasm_test', { shape: [4, 3], dtype: 'float32' });
  assert.deepEqual(made, { path: 'scratch/wasm_test', fresh: true }); ok();
  await assert.rejects(c.arrays.create(storePath, 'scratch/wasm_test', { shape: [4, 3], dtype: 'float32' }), (e: InstanceType<typeof NxrIoError>) => e.code === 'already_exists'); ok();
  await c.arrays.put(storePath, 'scratch/wasm_test', data, [[0, 4], [0, 3]]);
  assert.deepEqual(Array.from(await read(await openLocal(path.join(STORE, storePath)), 'scratch/wasm_test')), Array.from(data)); ok();
  // any region (read-modify-write of the chunks it touches): a 1x2 patch inside the array, read back by the Studio
  await c.arrays.put(storePath, 'scratch/wasm_test', Float32Array.from([7, 8]), [[1, 2], [1, 3]]);
  const patched = Array.from(data); patched[1 * 3 + 1] = 7; patched[1 * 3 + 2] = 8;
  assert.deepEqual(Array.from(await read(await openLocal(path.join(STORE, storePath)), 'scratch/wasm_test')), patched); ok();
  await assert.rejects(c.arrays.put(storePath, 'scratch/wasm_test', new Float64Array(12), [[0, 4], [0, 3]]), (e: InstanceType<typeof NxrIoError>) => e.code === 'invalid_argument'); ok();
  await assert.rejects(c.read(storePath, '../../d/dataset.sqlite'), (e: InstanceType<typeof NxrIoError>) => e.code === 'invalid_argument'); ok();
  await c.arrays.delete(storePath, 'scratch/wasm_test');
  await assert.rejects(c.read(storePath, 'scratch/wasm_test')); ok();
  await assert.rejects(c.imports.post({ dataset: 'd', source: { kind: 'store', path: '/x' } }), (e: InstanceType<typeof NxrIoError>) => e.code === 'unsupported'); ok();
}
c.close();
console.log(`wasm-io vs the Studio: ${checks} checks passed`);
