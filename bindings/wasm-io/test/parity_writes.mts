// The ROW WRITES of wasmClient against the Studio's own service (createModelService), on two copies of one synthetic datastore (make_store.py):
// the same calls through each, then the two databases compared table by table (ids equal by construction; times masked).
//   npx tsx <this> <studio checkout> <wasm-io build folder> <datastore>
import assert from 'node:assert/strict';
import { cpSync, readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import path from 'node:path';
import { copyKeys, keysOf, readKey } from './store_keys.mjs';   // the store's KEYS, at format 51 (files) and 52 (rows of the subject's small_files.sqlite)

const [, , STUDIO, PKG, STORE] = process.argv;
const { createModelService } = await import(path.join(STUDIO, 'backend/src/service/model.ts'));
const { wasmClient, NxrIoError } = await import(path.join(PKG, 'index.mjs'));

const A = `${STORE}-studio`, B = `${STORE}-wasm`;
cpSync(STORE, A, { recursive: true });
cpSync(STORE, B, { recursive: true });
const studio = createModelService();
await studio.open(A);
const wasm = await wasmClient(B, { program: 'cortical-flow' });   // stamped as the Studio's service stamps what it builds
let checks = 0;

const TIME = /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$/;
const deep = (v: unknown): unknown => typeof v === 'string' && TIME.test(v) ? '<T>' : Array.isArray(v) ? v.map(deep)
  : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, deep(x)])) : v;
const mask = (rows: Record<string, unknown>[]) => rows.map((r) => deep(r) as Record<string, unknown>);
const dbOf = (root: string, dataset?: string) => {
  const cat = JSON.parse(readFileSync(path.join(root, 'catalog.json'), 'utf8'));
  const d = dataset ? cat.datasets.find((x: { name: string }) => x.name === dataset) : cat.datasets[0];
  return new DatabaseSync(path.join(root, d.database), { readOnly: true });
};
// files the Studio keeps and nxr-neuro removes, once a known difference has made them (see the manifold deletes)
const knownOnlyInStudio: string[] = [];
// topologies the Studio's service leaves behind a deleted manifold and nxr-neuro (and nxr) delete with it: rows of `topology` / `topology_cell`
const knownTopologiesOnlyInStudio: string[] = [];
function same(what: string, onlyInStudio: readonly string[] = knownOnlyInStudio) {
  const names = (JSON.parse(readFileSync(path.join(A, 'catalog.json'), 'utf8')).datasets as { name: string }[]).map((d) => d.name);
  for (const n of names) sameDb(`${what} [${n}]`, dbOf(A, n), dbOf(B, n));
  const ca = JSON.parse(readFileSync(path.join(A, 'catalog.json'), 'utf8')), cb = JSON.parse(readFileSync(path.join(B, 'catalog.json'), 'utf8'));
  assert.deepEqual(cb.datasets.map((d: { name: string; path: string; database: string; subjects: number }) => [d.name, d.path, d.database, d.subjects]),
                   ca.datasets.map((d: { name: string; path: string; database: string; subjects: number }) => [d.name, d.path, d.database, d.subjects]), `${what}: catalog`);
  sameFiles(what, onlyInStudio);
  checks++;
}
function sameDb(what: string, a: InstanceType<typeof DatabaseSync>, b: InstanceType<typeof DatabaseSync>) {
  const tables = (a.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name").all() as { name: string }[]).map((t) => t.name);
  // the app's `measure` dictionary is minted by each side (its id differs): named, and every table's rows sorted, before comparing
  // and so are the spaces and coordinate systems a coordinate-system key creates: each minted id is named by what it is, everywhere
  const minted = (d: InstanceType<typeof DatabaseSync>) => {
    const m = new Map<string, string>();
    const md = (d.prepare("SELECT id FROM dictionary WHERE scope = 'app' AND name = 'measure'").get() as { id: string } | undefined)?.id;
    if (md) m.set(md, '<MEASURE-DICTIONARY>');
    for (const r of d.prepare('SELECT id, name FROM dataset').all() as { id: string; name: string }[]) m.set(r.id, `<dataset ${r.name}>`);
    for (const r of d.prepare("SELECT id, name FROM dictionary WHERE name = 'frequency bands'").all() as { id: string; name: string }[]) m.set(r.id, `<dictionary ${r.name}>`);
    for (const r of d.prepare('SELECT id, subject_id, name FROM space').all() as { id: string; subject_id: string; name: string }[]) m.set(r.id, `<space ${r.subject_id} ${r.name}>`);
    for (const r of d.prepare('SELECT c.id, s.subject_id, s.name AS space, c.name FROM coordinate_system c JOIN space s ON s.id = c.space_id').all() as { id: string; subject_id: string; space: string; name: string }[])
      m.set(r.id, `<system ${r.subject_id} ${r.space} ${r.name}>`);
    return m;
  };
  const ia = minted(a), ib = minted(b);
  const named = (rows: unknown[], ids: Map<string, string>) => {
    let t = JSON.stringify(rows);
    for (const [id, label] of ids) t = t.split(id).join(label);
    // this repository is public: its measure descriptions credit "(nxr-neuro)", where the Studio's text still names the decision and its date: the source tag alone is compared (and the wording of one remark on a trap)
    t = t.replace(/\(nxr-neuro, (?:PI )?\d{4}-\d\d-\d\d\)/g, '(nxr-neuro)').replace(/the (?:lab|project)'s trap/g, 'the trap');
    return (JSON.parse(t) as unknown[]).map((r) => JSON.stringify(r)).sort();
  };
  for (const t of tables) {
    // the datastore row names the folder it was opened in (the two copies sit in different folders)
    const q = t === 'datastore' ? 'SELECT id, driver, created_utc, created_by, modified_utc, modified_by FROM datastore ORDER BY 1' : `SELECT * FROM "${t}" ORDER BY 1`;
    let ra = JSON.parse(JSON.stringify(a.prepare(q).all())) as Record<string, unknown>[];
    if (t === 'topology' || t === 'topology_cell') {
      const gone = ra.filter((r) => knownTopologiesOnlyInStudio.includes(String(t === 'topology' ? r.id : r.topology_id)));
      const kept = JSON.parse(JSON.stringify(b.prepare(q).all())) as Record<string, unknown>[];
      for (const r of gone) assert.ok(!kept.some((k) => JSON.stringify(k) === JSON.stringify(r)), `${what}: ${t} ${JSON.stringify(r)} should be the Studio's alone`);
      ra = ra.filter((r) => !gone.includes(r));
    }
    assert.deepEqual(named(mask(JSON.parse(JSON.stringify(b.prepare(q).all()))), ib), named(mask(ra), ia), `${what}: table ${t}`);
  }
  a.close(); b.close();
}
function sameFiles(what: string, onlyInStudio: readonly string[]) {
  // the KEYS: every file and every small-file row of the two datastores (the database, its journal and lock files, and the catalog aside), by relative path
  const files = (root: string) => keysOf(root).filter((f) => !/dataset\.sqlite|catalog\.json|writer\.lock|\.lock$/.test(f));
  // a KNOWN difference, asserted rather than hidden: files the Studio keeps and nxr-neuro (with nxr) removes
  const fa = files(A);
  // (a key the Studio's own store no longer keeps is no difference at all: only a key in nxr-neuro's store is wrong)
  for (const f of onlyInStudio) assert.ok(!files(B).includes(f), `${what}: ${f} should be the Studio's alone`);
  const want = fa.filter((f) => !onlyInStudio.includes(f)), got = files(B);
  assert.deepEqual(got, want, `${what}: files; only in nxr-neuro's: ${JSON.stringify(got.filter((f) => !want.includes(f)))}; only in the Studio's: ${JSON.stringify(want.filter((f) => !got.includes(f)))}`);
  checks++;
}
same('as copied');
const subjects = dbOf(B).prepare('SELECT id, name FROM subject ORDER BY name').all() as { id: string; name: string }[];
const sub = subjects.find((s) => s.name === 'sub-1')!;

// subjects.put: the metadata; path and dataset never move
const patch = { name: 'sub-renamed', source_format: 'obj', path: 'elsewhere.nxr.zarr', dataset_id: 'x' };
const ea = studio.updateSubject(sub.id, patch as never, 'tester');
const wa = await wasm.subjects.put(sub.id, patch, 'tester');
assert.deepEqual(mask([wa])[0], mask([JSON.parse(JSON.stringify(ea))])[0]); checks++;
same('subjects.put');
await assert.rejects(wasm.subjects.put('no-such-id', {}), (e: InstanceType<typeof NxrIoError>) => e.code === 'no_such_row'); checks++;

// fields.post: SAVING A RESULT. The bytes first, identical on both sides (written once by wasmClient, copied), so the ROW logic is what is compared;
// then the row through each — the Studio's createField and wasmClient — and every table and the arrays' zarr.json (the drained attributes) compared.
{
  const storePath = `d/${(dbOf(B).prepare('SELECT path FROM subject WHERE id = ?').get(sub.id) as { path: string }).path}`;
  const oct = dbOf(B).prepare("SELECT id, n_vertices, n_faces FROM manifold WHERE subject_id = ? AND name = 'oct'").get(sub.id) as { id: string; n_vertices: number; n_faces: number };
  const put = async (p: string, data: Float32Array | Int32Array, shape: number[]) => {
    await wasm.arrays.create(storePath, p, { shape, dtype: data instanceof Int32Array ? 'int32' : 'float32' });
    await wasm.arrays.put(storePath, p, data, shape.map((n) => [0, n] as [number, number]));
    copyKeys(B, A, path.posix.join(storePath, p));
  };
  await put('source/oct_curv', Float32Array.from({ length: oct.n_vertices }, (_, i) => i / 10), [oct.n_vertices]);
  await put('source/oct_face_area', Float32Array.from({ length: oct.n_faces }, (_, i) => 1 + i), [oct.n_faces]);
  await put('source/oct_big', Int32Array.from({ length: oct.n_faces }, (_, i) => i % 2), [oct.n_faces]);
  await put('source/oct_wrong', new Float32Array(5), [5]);
  const dsubId = subjects.find((s) => s.name === '@default_subject')!.id;
  const f1 = { id: '01a1cc00-0000-7000-8000-000000000001', subject_id: sub.id, name: 'curvature', path: 'source/oct_curv', kind: 'curvature', manifold_id: oct.id,
               value_type: 'scalar', n_components: 1, unit: '1/m', function: 'mean curvature', producer_json: '{"by":"test"}' };
  const c1 = { contributions: [{ subject_id: dsubId, weight: 0.5 }] };
  const ea = await studio.createField(f1 as never, 'tester', c1 as never);
  const wa = await wasm.fields.post(f1, 'tester', c1);
  assert.deepEqual(mask([wa])[0], mask([JSON.parse(JSON.stringify(ea))])[0]); checks++;
  const f2 = { id: '01a1cc00-0000-7000-8000-000000000002', subject_id: sub.id, name: 'face area', path: 'source/oct_face_area', kind: 'area', manifold_id: oct.id, cell: '2',
               value_type: 'scalar', n_components: 1, unit: 'm2' };
  const c2 = { selections: [{ id: '01a1cc00-0000-7000-8000-000000000003', name: 'big faces', path: 'source/oct_big', type: 'indices', manifold_id: oct.id, cell: '2', array_path: 'source/oct_big', n_elements: oct.n_faces }] };
  await studio.createField(f2 as never, 'tester', c2 as never);
  await wasm.fields.post(f2, 'tester', c2);
  same('fields.post');
  for (const p of ['source/oct_curv', 'source/oct_face_area', 'source/oct_big']) {
    const za = JSON.parse(readKey(A, `${storePath}/${p}/zarr.json`).toString('utf8')), zb = JSON.parse(readKey(B, `${storePath}/${p}/zarr.json`).toString('utf8'));
    assert.deepEqual(mask([zb])[0], mask([za])[0], `${p}: zarr.json`); checks++;
  }
  // refused alike: a value count that does not fit the cells; a cell above the manifold's rank; no path
  for (const [bad, why] of [[{ ...f1, id: undefined, name: 'wrong', path: 'source/oct_wrong' }, /holds 5 values/], [{ ...f1, id: undefined, name: 'cell3', cell: '3' }, /rank 3 above|does not know/],
                            [{ ...f1, id: undefined, name: 'nopath', path: undefined }, /needs a path/]] as const) {
    await assert.rejects(Promise.resolve().then(() => studio.createField(bad as never, 'tester')), why);
    await assert.rejects(wasm.fields.post(bad, 'tester'), why); checks++;
  }
  same('fields.post refusals');
}

// SELECTIONS and MEASUREMENTS: the Studio's repositories against wasmClient
{
  const repos = studio.repos();
  const oct = dbOf(B).prepare("SELECT id FROM manifold WHERE subject_id = ? AND name = 'oct'").get(sub.id) as { id: string };
  const s1 = { id: '01a1cc00-0000-7000-8000-000000000010', subject_id: sub.id, name: 'drawn', path: 'source/drawn', type: 'indices', manifold_id: oct.id, n_elements: 3, label: 'a drawn patch' };
  const ch1 = { extents: [{ axis: 0, start: 0, stop: 3 }], measurements: [{ measure: 'area', value: 1.5, unit: 'm²', computed_by: 'tester' }] };
  const e1 = await repos.selections.create(s1 as never, ch1 as never, 'tester');
  const w1 = await wasm.selections.post(s1, ch1, 'tester');
  assert.deepEqual(mask([w1])[0], mask([JSON.parse(JSON.stringify(e1))])[0]); checks++;
  const s1b = { ...JSON.parse(JSON.stringify(e1)), label: 'renamed', description: 'edited' };
  await repos.selections.update(s1b as never, { spans: [{ start: 0, stop: 2 }, { start: 4, stop: 6, code: 1 }] } as never, 'tester');
  await wasm.selections.put(s1b, { spans: [{ start: 0, stop: 2 }, { start: 4, stop: 6, code: 1 }] }, 'tester');
  same('selections post/put');
  await assert.rejects(wasm.selections.put({ ...s1b, id: 'nope' }, undefined, 'tester'), (e: InstanceType<typeof NxrIoError>) => e.code === 'no_such_row'); checks++;
  await assert.rejects(wasm.selections.post({ ...s1, id: undefined, name: 'bad', cell: '3' }, undefined, 'tester'), /rank 3 above/); checks++;

  // member measurements on the group tree's level 2 (codes h·4 + t), the selection's own, a derived one refused
  const lvl = dbOf(B).prepare("SELECT id FROM selection WHERE name = 'cortex_pial tiles L2'").get() as { id: string };
  const rows = [] as Record<string, unknown>[];
  for (let code = 0; code < 8; code++) {
    rows.push({ selection_id: lvl.id, code, measure: 'mean', value: code * 0.5, weight: 1 + code });
    rows.push({ selection_id: lvl.id, code, measure: 'mean_square', value: code * code * 0.5 + 1, weight: 1 + code });
    rows.push({ selection_id: lvl.id, code, measure: 'max', value: code + 2, computed_utc: '2026-01-01T00:00:00.000Z' });
  }
  rows.push({ selection_id: lvl.id, measure: 'area', value: 0.01 });
  assert.equal(await repos.measurements.create(rows as never, 'tester'), rows.length);
  assert.equal(await wasm.measurements.post(rows, 'tester'), rows.length); checks++;
  same('measurements.post');
  await assert.rejects(Promise.resolve().then(() => repos.measurements.create([{ selection_id: lvl.id, code: 0, measure: 'variance', value: 1 }] as never)), /DERIVED/);
  await assert.rejects(wasm.measurements.post([{ selection_id: lvl.id, code: 0, measure: 'variance', value: 1 }]), /DERIVED/); checks++;
  const norm = (x: unknown) => JSON.parse(JSON.stringify(x, (_k, v) => (typeof v === 'number' && Number.isNaN(v) ? null : v)));
  assert.deepEqual(norm(await wasm.measurements.get(undefined, lvl.id)), norm(await repos.measurements.kinds(lvl.id)), 'kinds'); checks++;
  for (const q of [{ selectionId: lvl.id, measure: 'mean' }, { selectionId: lvl.id, measure: 'mean', shift: 1, order: 'asc' }, { selectionId: lvl.id, measure: 'max', minValue: 4, limit: 3 },
                   { selectionId: lvl.id, measure: 'variance' }, { selectionId: lvl.id, measure: 'sd', shift: 2, maxValue: 3 }, { selectionId: lvl.id, measure: 'rms', ofFieldId: null }]) {
    assert.deepEqual(norm(await wasm.measurements.get(q)), norm(await repos.measurements.query(q as never)), `query ${JSON.stringify(q)}`); checks++;
  }
  assert.equal(await wasm.measurements.delete({ selectionId: lvl.id, measure: 'max' }), await repos.measurements.delete({ selectionId: lvl.id, measure: 'max' } as never)); checks++;
  await repos.selections.delete(s1.id);
  await wasm.selections.delete(s1.id);
  same('selections.delete, measurements.delete');
}

// OPERATORS, and a field made BY one FROM an input (the step check, end to end); fields.put / delete
{
  const repos = studio.repos();
  const storePath = `d/${(dbOf(B).prepare('SELECT path FROM subject WHERE id = ?').get(sub.id) as { path: string }).path}`;
  const oct = dbOf(B).prepare("SELECT id FROM manifold WHERE subject_id = ? AND name = 'oct'").get(sub.id) as { id: string };
  const put = async (p: string, data: Float32Array, shape: number[]) => {
    await wasm.arrays.create(storePath, p, { shape, dtype: 'float32' });
    await wasm.arrays.put(storePath, p, data, shape.map((n) => [0, n] as [number, number]));
    copyKeys(B, A, path.posix.join(storePath, p));
  };
  await put('source/oct_smooth', Float32Array.from({ length: 36 }, (_, i) => (i % 7 === 0 ? 0.5 : 0.1)), [6, 6]);
  await put('source/oct_smoothed', new Float32Array(6), [6]);
  const op1 = { id: '01a1cc00-0000-7000-8000-000000000020', subject_id: sub.id, name: 'smooth', path: 'source/oct_smooth', kind: 'smoothing', from_manifold_id: oct.id, to_manifold_id: oct.id,
                from_value_type: 'scalar', to_value_type: 'scalar', layout: 'dense', n_rows: 6, n_cols: 6, method: 'heat', params_json: '{"t":1}' };
  const e1 = await studio.createOperator(op1 as never, 'tester');
  const w1 = await wasm.operators.post(op1, undefined, 'tester');
  assert.deepEqual(mask([w1])[0], mask([JSON.parse(JSON.stringify(e1))])[0]); checks++;
  const op2 = { id: '01a1cc00-0000-7000-8000-000000000021', subject_id: sub.id, name: 'gradient', kind: 'differential', from_manifold_id: oct.id, to_manifold_id: oct.id,
                from_value_type: 'scalar', to_value_type: 'vector3', layout: 'analytic', method: 'cotan', n_rows: 6, n_cols: 6 };
  await studio.createOperator(op2 as never, 'tester');
  await wasm.operators.post(op2, undefined, 'tester');
  const op3 = { id: '01a1cc00-0000-7000-8000-000000000022', subject_id: sub.id, name: 'smooth then gradient', kind: 'composed', from_manifold_id: oct.id, to_manifold_id: oct.id,
                from_value_type: 'scalar', to_value_type: 'vector3', layout: 'analytic', method: 'composition', n_rows: 6, n_cols: 6 };
  const f3 = { factors: [{ factor_operator_id: op1.id }, { factor_operator_id: op2.id }] };
  await studio.createOperator(op3 as never, 'tester', f3 as never);
  await wasm.operators.post(op3, f3, 'tester');
  same('operators.post');
  // refused alike: an inverse whose value types are not swapped; a composition that does not chain; a dense array of another shape
  const badInv = { ...op2, id: undefined, name: 'bad inverse', inverts_id: op2.id, inverse_kind: 'exact' };
  await assert.rejects(Promise.resolve().then(() => studio.createOperator(badInv as never, 'tester')), /value types are not swapped/);
  await assert.rejects(wasm.operators.post(badInv, undefined, 'tester'), /value types are not swapped/); checks++;
  const badChain = { factors: [{ factor_operator_id: op2.id }, { factor_operator_id: op1.id }] };
  await assert.rejects(Promise.resolve().then(() => studio.createOperator({ ...op3, id: undefined, name: 'bad chain' } as never, 'tester', badChain as never)), /reads scalar, gradient writes vector3/);
  await assert.rejects(wasm.operators.post({ ...op3, id: undefined, name: 'bad chain' }, badChain, 'tester'), /reads scalar, gradient writes vector3/); checks++;
  await assert.rejects(Promise.resolve().then(() => studio.createOperator({ ...op1, id: undefined, name: 'bad shape', n_rows: 5 } as never, 'tester')), /stored array is 6×6/);
  await assert.rejects(wasm.operators.post({ ...op1, id: undefined, name: 'bad shape', n_rows: 5 }, undefined, 'tester'), /stored array is 6×6/); checks++;
  // a field made BY `smooth` FROM the curvature: the step meets; one the operator cannot make is refused
  const fs1 = { id: '01a1cc00-0000-7000-8000-000000000023', subject_id: sub.id, name: 'smoothed curvature', path: 'source/oct_smoothed', kind: 'curvature', manifold_id: oct.id,
                value_type: 'scalar', n_components: 1, by_operator_id: op1.id, derived_from_id: '01a1cc00-0000-7000-8000-000000000001' };
  await studio.createField(fs1 as never, 'tester');
  await wasm.fields.post(fs1, 'tester');
  const fsBad = { ...fs1, id: undefined, name: 'wrong type', value_type: 'vector3', n_components: 1 };
  await assert.rejects(Promise.resolve().then(() => studio.createField(fsBad as never, 'tester')), /smooth writes scalar, the field is vector3/);
  await assert.rejects(wasm.fields.post(fsBad, 'tester'), /smooth writes scalar, the field is vector3/); checks++;
  same('a field made by an operator');
  // an array a row names is not deleted directly (D164: it goes with its row); one no row names is
  await assert.rejects(wasm.arrays.delete(storePath, 'source/oct_curv'), /named by/); checks++;
  // updates and deletes
  const op1b = { ...JSON.parse(JSON.stringify(e1)), description: 'heat smoothing, t = 1' };
  await repos.operators.update(op1b as never, 'tester');
  await wasm.operators.put(op1b, 'tester');
  const fb = { ...JSON.parse(JSON.stringify(repos.fields.read(fs1.id))), label: 'smoothed' };
  await repos.fields.update(fb as never, 'tester');
  await wasm.fields.put(fb, 'tester');
  same('operators.put, fields.put');
  await repos.fields.delete(fs1.id); await wasm.fields.delete(fs1.id);
  await repos.operators.delete(op3.id); await wasm.operators.delete(op3.id);
  same('fields.delete, operators.delete');
}

// MANIFOLDS: a surface (stored geometry, a coordinate-system key), a time line and an eigen line (closed form; a new space of two systems), a product of two,
// an update, the deletes (the surface's arrays with it), and the refusals
{
  const repos = studio.repos();
  const storePath = `d/${(dbOf(B).prepare('SELECT path FROM subject WHERE id = ?').get(sub.id) as { path: string }).path}`;
  const put = async (p: string, data: Float32Array | Int32Array, shape: number[]) => {
    await wasm.arrays.create(storePath, p, { shape, dtype: data instanceof Int32Array ? 'int32' : 'float32' });
    await wasm.arrays.put(storePath, p, data, shape.map((n) => [0, n] as [number, number]));
    copyKeys(B, A, path.posix.join(storePath, p));
  };
  await put('source/oct2/positions', Float32Array.from([1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1].map((x) => x * 0.02)), [6, 3]);
  await put('source/oct2/faces', Int32Array.from([0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5]), [8, 3]);
  const id = (n: number) => `01a1cc00-0000-7000-8000-0000000001${String(n).padStart(2, '0')}`;
  // every manifold column, as the Studio's callers give them (its service cannot bind an absent one)
  const COLS = { acquisition_id: null, component_of_id: null, component_ordinal: null, from_index: null, derived_from_id: null, operator: null, comment: null, source: null,
                 producer_json: null, label: null, topology_id: null, geometry_id: null, created_by: null, modified_by: null };
  const surf = { ...COLS, id: id(1), subject_id: sub.id, name: 'oct2', path: 'source/oct2', type: 'surface', is_primary: 0, n_vertices: 6, n_edges: 12, n_faces: 8, n_volumes: null, label: 'a second octahedron' };
  const surfC = { topology: { id: id(2), kind: 'simplicial', rank: 2, n_components: 1, winding: 'ccw-outward', closed: 1, orientable: 1, cells: [{ rank: 2, path: 'source/oct2/faces' }] },
                  geometry: { id: id(3), form: 'stored', coordinate_system_key: 'scs', positions_path: 'source/oct2/positions', direction_json: null } };
  const es = await studio.createManifold(surf as never, surfC as never, 'tester');
  const ws = await wasm.manifolds.post(surf, surfC, 'tester');
  assert.deepEqual(mask([ws])[0], mask([JSON.parse(JSON.stringify(es))])[0]); checks++;
  const line = { ...COLS, id: id(4), subject_id: sub.id, name: 'time', path: 'time/clock', type: 'line', is_primary: 0, n_vertices: 100, n_edges: 99, n_faces: null, n_volumes: null };
  const lineC = { topology: { id: id(5), kind: 'path', rank: 1, n_components: 1 },
                  geometry: { id: id(6), form: 'closed_form', coordinate_system_key: 'clock', positions_path: null, direction_json: null,
                              dimensions: [{ ordinal: 0, name: 't', n_vertices: 100, spacing: 0.01, origin: 0, unit: 's', manifold_id: null }] } };
  await studio.createManifold(line as never, lineC as never, 'tester');
  await wasm.manifolds.post(line, lineC, 'tester');
  const eig = { ...COLS, id: id(7), subject_id: sub.id, name: 'hodgeDirac modes', path: 'eigenvalue/hodgeDirac', type: 'line', is_primary: 0, n_vertices: 40, n_edges: 39, n_faces: null, n_volumes: null };
  const eigC = { topology: { id: id(8), kind: 'path', rank: 1, n_components: 1 },
                 geometry: { id: id(9), form: 'closed_form', coordinate_system_key: 'eigenvalue:hodgeDirac', positions_path: null, direction_json: null,
                             dimensions: [{ ordinal: 0, name: 'mode', n_vertices: 40, spacing: 1, origin: 0, unit: null, manifold_id: null }] } };
  await studio.createManifold(eig as never, eigC as never, 'tester');
  await wasm.manifolds.post(eig, eigC, 'tester');
  // the same key again: the space and its systems are found, not made twice
  const eig2 = { ...eig, id: id(10), name: 'hodgeDirac modes 2', path: 'eigenvalue/hodgeDirac2' };
  const eig2C = { topology: { ...eigC.topology, id: id(11) }, geometry: { ...eigC.geometry, id: id(12) } };
  await studio.createManifold(eig2 as never, eig2C as never, 'tester');
  await wasm.manifolds.post(eig2, eig2C, 'tester');
  const prod = { id: id(13), subject_id: sub.id, type: 'product', topology_id: id(14), geometry_id: id(15) };
  const ep = await studio.createManifold(prod as never, { factors: [surf.id, line.id] } as never, 'tester');
  const wp = await wasm.manifolds.post(prod, { factors: [surf.id, line.id] }, 'tester');
  assert.deepEqual(mask([wp])[0], mask([JSON.parse(JSON.stringify(ep))])[0]); checks++;
  same('manifolds.post');
  await assert.rejects(Promise.resolve().then(() => studio.createManifold({ ...prod, id: undefined, topology_id: undefined, geometry_id: undefined } as never, { factors: [surf.id] } as never)), /at least two factors/);
  await assert.rejects(wasm.manifolds.post({ ...prod, id: undefined, topology_id: undefined, geometry_id: undefined }, { factors: [surf.id] }), /at least two factors/); checks++;
  await assert.rejects(Promise.resolve().then(() => studio.createManifold({ ...surf, id: undefined, name: 'bare' } as never, undefined)), /needs a topology/);
  await assert.rejects(wasm.manifolds.post({ ...surf, id: undefined, name: 'bare' }), /needs a topology/); checks++;
  const sb = { ...JSON.parse(JSON.stringify(repos.manifolds.read(surf.id))), label: 'relabelled' };
  for (const k of ['topology', 'geometry', 'dimensions']) delete sb[k];
  await repos.manifolds.update(sb as never, 'tester');
  await wasm.manifolds.put(sb, 'tester');
  same('manifolds.put');
  await repos.manifolds.delete(prod.id); await wasm.manifolds.delete(prod.id);
  await repos.manifolds.delete(surf.id); await wasm.manifolds.delete(surf.id);
  // KNOWN DIFFERENCE (D164): the Studio's named paths count a topology's cells while the TOPOLOGY row stands, so the faces of a deleted surface (and its folder's
  // zarr.json) stay on disk under a topology no manifold uses; nxr's and nxr-neuro's count them only while a MANIFOLD stands on them ("one rule with the
  // Studio", nxr/io/named_paths.py), so they go. The Studio is the one out of step (backend/src/db/named-paths.ts).
  // and the topologies themselves: the Studio's service keeps a deleted manifold's topology row; nxr-neuro deletes it when no manifold stands on it
  knownTopologiesOnlyInStudio.push(id(2), id(14));
  knownOnlyInStudio.push('d/sub-1.nxr.zarr/source/oct2/faces/c.0.0', 'd/sub-1.nxr.zarr/source/oct2/faces/c/0/0', 'd/sub-1.nxr.zarr/source/oct2/faces/zarr.json', 'd/sub-1.nxr.zarr/source/oct2/zarr.json');
  same('manifolds.delete (the surface\'s arrays with it)');
}

// DICTIONARIES and JOBS
{
  const repos = studio.repos();
  const oct = dbOf(B).prepare("SELECT id FROM manifold WHERE subject_id = ? AND name = 'oct'").get(sub.id) as { id: string };
  const id = (n: number) => `01a1cc00-0000-7000-8000-0000000002${String(n).padStart(2, '0')}`;
  const dict = { id: id(1), name: 'regions', scope: 'subject', dataset_id: null, subject_id: sub.id, description: 'two regions' };
  const entries = { entries: [{ code: 0, name: 'unassigned', color_json: '[0,0,0]', measurement: null, measurement_unit: null, function: null, params_json: null, attributes_json: null, ordinal: 0 },
                              { code: 1, name: 'north', color_json: '[200,10,10]', measurement: 1.5, measurement_unit: 'm²', function: null, params_json: null, attributes_json: '{"k":1}', ordinal: 1 }] };
  const ed = await repos.dictionaries.create(dict as never, entries as never, 'tester');
  const wd = await wasm.dictionaries.post(dict, entries, 'tester');
  assert.deepEqual(mask([wd])[0], mask([JSON.parse(JSON.stringify(ed))])[0]); checks++;
  const storePath = `d/${(dbOf(B).prepare('SELECT path FROM subject WHERE id = ?').get(sub.id) as { path: string }).path}`;
  await wasm.arrays.create(storePath, 'source/oct_regions', { shape: [6], dtype: 'int32' });
  await wasm.arrays.put(storePath, 'source/oct_regions', Int32Array.from([0, 1, 1, 0, 1, 0]), [[0, 6]]);
  copyKeys(B, A, path.posix.join(storePath, 'source/oct_regions'));
  const sel = { id: id(2), subject_id: sub.id, name: 'regions', path: 'source/oct_regions', type: 'set', manifold_id: oct.id, dictionary_id: dict.id, array_path: 'source/oct_regions', n_elements: 6, n_members: 2 };
  await repos.selections.create(sel as never, undefined, 'tester');
  await wasm.selections.post(sel, undefined, 'tester');
  const app = { id: id(3), name: 'app labels', scope: 'app', dataset_id: null, subject_id: null, description: 'app-wide' };
  await repos.dictionaries.create(app as never, { entries: [{ ...entries.entries[1], code: 7 }] } as never, 'tester');
  await wasm.dictionaries.post(app, { entries: [{ ...entries.entries[1], code: 7 }] }, 'tester');
  same('dictionaries.post');
  const dictB = { ...JSON.parse(JSON.stringify(ed)), description: 'renamed' };
  delete dictB.entries;
  await repos.dictionaries.update(dictB as never, { entries: [{ ...entries.entries[0] }, { ...entries.entries[1], name: 'south' }] } as never, 'tester');
  await wasm.dictionaries.put(dictB, { entries: [{ ...entries.entries[0] }, { ...entries.entries[1], name: 'south' }] }, 'tester');
  await repos.dictionaries.update({ ...dictB, description: 'kept entries' } as never, undefined, 'tester');
  await wasm.dictionaries.put({ ...dictB, description: 'kept entries' }, undefined, 'tester');
  same('dictionaries.put (the selection reading it touched; entries kept when none are given)');
  await assert.rejects(repos.dictionaries.delete(dict.id), /FOREIGN KEY|constraint/i);
  await assert.rejects(wasm.dictionaries.delete(dict.id), /FOREIGN KEY|constraint/i); checks++;
  await repos.selections.delete(sel.id); await wasm.selections.delete(sel.id);
  await repos.dictionaries.delete(dict.id); await wasm.dictionaries.delete(dict.id);
  same('dictionaries.delete');

  const job = { id: id(4), kind: 'opticalflow', host: 'main', dataset_id: null, subject_id: sub.id, inputs_json: '{"field":"x"}', params_json: '{"alpha":1}' };
  const ej = repos.jobs.create(job as never, 'tester');
  const wj = await wasm.jobs.post(job, 'tester');
  assert.deepEqual(mask([wj])[0], mask([JSON.parse(JSON.stringify(ej))])[0]); checks++;
  repos.jobs.update({ id: job.id, status: 'running', progress: 0.5 } as never); await wasm.jobs.put({ id: job.id, status: 'running', progress: 0.5 });
  repos.jobs.update({ id: job.id, status: 'done', result_json: '{"ok":true}' } as never); await wasm.jobs.put({ id: job.id, status: 'done', result_json: '{"ok":true}' });
  same('jobs.post/put');
  const got = await wasm.jobs.get(job.id) as Record<string, unknown>;
  assert.equal(got.status, 'done'); assert.ok(TIME.test(String(got.started_utc)) && TIME.test(String(got.finished_utc))); checks++;
  assert.deepEqual(mask(await wasm.jobs.get(undefined, { subjectId: sub.id }) as never), mask(JSON.parse(JSON.stringify(repos.jobs.list({ subjectId: sub.id }))))); checks++;
  assert.equal(await wasm.jobs.get('no-such-job'), null); checks++;
}

// LAYOUT: where a node goes — the same answer, or the same refusal, for every manifold and every kind about to be written
{
  const storePath = `d/${(dbOf(B).prepare('SELECT path FROM subject WHERE id = ?').get(sub.id) as { path: string }).path}`;
  const ds = dbOf(B);
  const mine = (ds.prepare('SELECT id FROM manifold WHERE subject_id = ?').all(sub.id) as { id: string }[]).map((r) => r.id);
  const theirs = (ds.prepare('SELECT id FROM manifold WHERE subject_id <> ? LIMIT 1').all(sub.id) as { id: string }[]).map((r) => r.id);
  const qs: Record<string, unknown>[] = [
    ...mine.map((m) => ({ name: 'node', manifold: m })),
    { name: 'x', manifold: theirs[0] }, { name: 'x', manifold: 'no-such-manifold' }, { name: 'bad name!', manifold: mine[0] }, { name: 'x' },
    { name: 'x', type: 'surface', manifoldName: 'cortex_pial' }, { name: 'x', type: 'surface', manifoldName: 'head' }, { name: 'x', type: 'volume' },
    { name: 'x', type: 'line', role: 'time' }, { name: 'x', type: 'line', role: 'spectral' }, { name: 'x', type: 'line', role: 'eigen' }, { name: 'x', type: 'line' },
    { name: 'x', type: 'points' }, { name: 'x', type: 'points', operator: 'hodgeDirac' }, { name: 'x', type: 'curve' }, { name: 'x', type: 'graph' },
    { name: 'x', type: 'product', factors: [mine[0], mine[0]] }, { name: 'x', type: 'product', factors: ['points:sensor', 'line:time'] },
    { name: 'x', type: 'product', factors: ['points:operator', 'line:eigen'] }, { name: 'x', type: 'product' },
  ];
  for (const q of qs) {
    let want: unknown, got: unknown;
    try { want = { ok: studio.layout(storePath, q as never) }; } catch (e) { want = { error: (e as Error).message }; }
    try { got = { ok: await wasm.layout(storePath, q as never) }; } catch (e) { got = { error: (e as Error).message }; }
    assert.deepEqual(got, want, `layout ${JSON.stringify(q)}`); checks++;
  }
  let notSub: unknown;
  try { studio.layout('d/nobody.nxr.zarr', { name: 'x', type: 'volume' } as never); } catch (e) { notSub = (e as Error).message; }
  await assert.rejects(wasm.layout('d/nobody.nxr.zarr', { name: 'x', type: 'volume' }), (e: Error) => e.message === notSub); checks++;
}

// DATASETS: create (idempotent by name), refusals, delete — every database, file and the catalog compared
{
  const source = { tool: 'brainstorm', protocol: 'TutorialOmega', path: '/brainstorm_db/TutorialOmega' };
  const ed = await studio.ensureDataset('d2', source);
  const wd = await wasm.datasets.create('d2', source);
  const maskId = (r: Record<string, unknown>) => ({ ...(mask([JSON.parse(JSON.stringify(r))])[0] as Record<string, unknown>), id: '<ID>' });
  assert.deepEqual(maskId(wd), maskId(ed as never)); checks++;
  same('datasets.create');
  assert.equal((await wasm.datasets.create('d2')).id, wd.id, 'idempotent by name'); checks++;
  for (const bad of ['', 'a/b', '..', 'x\\y']) {
    let want: string | undefined;
    try { await studio.ensureDataset(bad); } catch (e) { want = (e as Error).message; }
    await assert.rejects(wasm.datasets.create(bad), (e: Error) => e.message === want, `create ${JSON.stringify(bad)}`); checks++;
  }
  assert.deepEqual((await wasm.datasets.catalog()).datasets.map((d: { name: string }) => d.name), ['d', 'd2']); checks++;
  studio.removeDataset('d2');
  assert.deepEqual(await wasm.datasets.delete('d2'), { deleted: 'd2' });
  same('datasets.delete');
  studio.removeDataset('d2');
  assert.deepEqual(await wasm.datasets.delete('d2'), { deleted: null }, 'one not there is not an error'); checks++;
  for (const bad of ['', '../x']) {
    let want: string | undefined;
    try { studio.removeDataset(bad); } catch (e) { want = (e as Error).message; }
    await assert.rejects(wasm.datasets.delete(bad), (e: Error) => e.message === want, `delete ${JSON.stringify(bad)}`); checks++;
  }
}

// subjects.delete: the default subject (selections, surfaces, frames, the group tree's measurements) — everything under it goes
const dsub = subjects.find((s) => s.name === '@default_subject')!;
studio.removeSubject(dsub.id);
assert.deepEqual(await wasm.subjects.delete(dsub.id), { id: dsub.id }); checks++;
same('subjects.delete');
const left = dbOf(B);
for (const t of ['manifold', 'selection', 'field', 'selection_measurement', 'geometry', 'topology'])
  assert.equal((left.prepare(`SELECT count(*) AS n FROM ${t} t ${t === 'selection_measurement' ? 'JOIN selection s ON s.id = t.selection_id WHERE s.subject_id' : 'WHERE t.subject_id'} = ?`).get(dsub.id) as { n: number }).n, 0, `${t} left behind`);
checks++;
await assert.rejects(wasm.subjects.delete(dsub.id), (e: InstanceType<typeof NxrIoError>) => e.code === 'no_such_row'); checks++;
wasm.close();
console.log(`wasm-io writes vs the Studio's service: ${checks} checks passed`);
