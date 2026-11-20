// The dataset API in Node (docs/specs/2026-11-19-io-service.md, step 5): the command line's commands as functions on synthetic datasets made in the test. Datasets, subjects and
// packages are created, listed and shown; every class is added from typed arrays and read back byte for byte in the input's precision; a pipeline is compiled and run; the writer
// lock of a handle refuses another process's write until it is closed. Mirrors bindings/python/tests/test_dataset_api.py. Synthetic data only.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { skip, load, PACKAGE } from './helpers.mjs';

const SFREQ = 100, SECONDS = 40, CHANNELS = 3;
const bytes = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
const same = (a, b) => assert.ok(bytes(a).equals(bytes(b)), `${a} is not ${b}`);
const scratch = () => mkdtempSync(join(process.env.NXR_TEST_TMP ?? tmpdir(), 'nxr-wasm-'));
async function demo() {
  const lib = await load();
  const dir = scratch();
  return { ...lib, dir, demo: lib.syntheticDataset(lib.nxr, dir, { name: 'demo', subject: 'S1', recording: 'rest', channels: CHANNELS, sfreq: SFREQ, seconds: SECONDS }), done: () => rmSync(dir, { recursive: true, force: true }) };
}

test('datasets, subjects and packages', { skip }, async () => {
  const { nxr, NxrError } = await load();
  const dir = scratch();
  const done = () => rmSync(dir, { recursive: true, force: true });
  try {
    const ds = join(dir, 'study');
    const made = await nxr.create.dataset({ dataset: ds });
    assert.equal(made.arrays, undefined);
    await nxr.create.subject({ dataset: ds, name: 'S1' });
    await nxr.create.subject({ dataset: ds, name: 'S2' });
    assert.deepEqual((await nxr.list.subjects({ dataset: ds })).subjects.map((s) => s.name), ['S1', 'S2']);
    assert.deepEqual((await nxr.list.subjects({ dataset: ds, subjects: ['S2'] })).subjects.map((s) => s.name), ['S2']);
    assert.ok(await nxr.show.subject({ dataset: ds, subject: 'S1' }));
    assert.ok(await nxr.show.dataset({ dataset: ds }));
    assert.deepEqual((await nxr.list.members({ dataset: ds })).subjects.map((m) => m.name), ['S1', 'S2']);
    await assert.rejects(nxr.create.subject({ dataset: ds, name: 'S1' }), (e) => e instanceof NxrError && e.code === 'already_exists');
    await assert.rejects(nxr.create.subject({ dataset: ds }), (e) => e.code === 'invalid_argument');
    // a package of its own, with a synthetic recording (progress lines as objects)
    const pkg = join(dir, 'P1.nxr.zarr');
    const p = await nxr.create.package({ package: pkg, name: 'P1', home_dataset: 'study' });
    assert.ok(p.subject_id);
    const lines = [];
    await nxr.create.recording({ package: pkg, synthetic: true, name: 'rest', channels: 2, sfreq: 200, duration: 4, frequency: [10, 12], seed: 3, progress: (l) => lines.push(l) });
    assert.ok(lines.length && lines.every((l) => typeof l === 'object'));
    assert.equal((await nxr.show.package({ package: pkg })).record.name, 'P1');
    assert.equal((await nxr.verify.package({ package: pkg })).ok, true);
    await assert.rejects(nxr.create.recording({ package: pkg, name: 'other' }), (e) => e.code === 'invalid_argument' && /synthetic/.test(e.message));
  } finally { done(); }
});

test('read subject and list each class', { skip }, async () => {
  const { nxr, demo: D, done } = await demo();
  try {
    const sub = await nxr.read.subject({ dataset: D, subject: 'S1' });
    assert.equal(sub.schema, 'nxr-neuro-subject@1');
    const rest = sub.fields.find((f) => f.name === 'rest');
    assert.equal(rest.type.sfreq, SFREQ);
    const keys = (x) => (Array.isArray(x) ? x.flatMap(keys) : x && typeof x === 'object' ? Object.entries(x).flatMap(([k, v]) => [k, ...keys(v)]) : []);
    assert.deepEqual(keys(sub).filter((k) => k.endsWith('_id') || k.endsWith('_json')), []);
    assert.equal('manifolds' in (await nxr.read.subject({ dataset: D, subject: 'S1', classes: ['fields'] })), false);
    const first = await nxr.list.manifolds({ dataset: D, subject: 'S1', limit: 1 });
    assert.equal(first.rows.length, 1);
    const more = await nxr.list.manifolds({ dataset: D, subject: 'S1', after: first.next });
    assert.deepEqual([...first.rows, ...more.rows].map((r) => r.name), sub.manifolds.map((m) => m.name));
    assert.deepEqual((await nxr.list.fields({ dataset: D, subject: 'S1', name: 'res', columns: 'name' })).rows, [{ name: 'rest' }]);
    assert.equal((await nxr.show.dataset({ dataset: D })).nodes.fields, sub.fields.length);
  } finally { done(); }
});

test('read each class; a time range is required', { skip }, async () => {
  const { nxr, NxrError, demo: D, done } = await demo();
  try {
    await assert.rejects(nxr.read.field({ dataset: D, subject: 'S1', name: 'rest' }), (e) => e instanceof NxrError && e.code === 'invalid_argument' && /a time range is required/.test(e.message));
    const f = await nxr.read.field({ dataset: D, subject: 'S1', name: 'rest', from: 0, to: 1 });
    assert.equal(f.class, 'field');
    assert.deepEqual(f.arrays.values.shape, [CHANNELS, 100]);
    assert.equal(f.arrays.values.dtype, 'float32');
    assert.ok(f.arrays.values instanceof Float32Array);
    const whole = await nxr.read.field({ dataset: D, subject: 'S1', name: 'rest', all: true, rows: [2, 0] });
    assert.deepEqual(whole.arrays.values.shape, [2, SECONDS * SFREQ]);
    same(whole.arrays.values.slice(0, 100), f.arrays.values.slice(2 * 100, 3 * 100));
    assert.deepEqual((await nxr.read.field({ dataset: D, subject: 'S1', name: 'rest', shape_only: true })).arrays, []);   // no array comes back
    const m = await nxr.read.manifold({ dataset: D, subject: 'S1', name: 'sensors' });
    assert.deepEqual(m.arrays.positions.shape, [CHANNELS, 3]);
    assert.deepEqual(m.names, ['C0', 'C1', 'C2']);
    const q = await nxr.read.manifold({ dataset: D, subject: 'S1', name: 'sensors', family: 'quality' });
    const names = Object.fromEntries(q.dictionary.map((e) => [e.code, e.name]));
    assert.deepEqual(Array.from(q.arrays.codes, (c) => names[c]), ['good', 'good', 'bad']);
    const b = await nxr.read.selection({ dataset: D, subject: 'S1', name: 'rest_bad_segments' });
    assert.deepEqual(Array.from(b.arrays.start), [0.3 * SECONDS]);
    assert.deepEqual(Array.from(b.arrays.stop), [0.375 * SECONDS]);
    await assert.rejects(nxr.read.statistic({ dataset: D, subject: 'S1', name: 'nothing' }), (e) => e.code === 'no_such_row');
    // read dataset: every array the rows name; one array by subject and path
    const listing = await nxr.read.dataset(D);
    assert.ok(listing.subjects.length && listing.arrays.length);
    const one = listing.arrays[0];
    const got = await nxr.read.dataset(D, { array: [one.subject, one.path] });
    const arr = Object.values(got.arrays)[0];
    assert.deepEqual(arr.shape, one.shape);
    assert.equal(arr.dtype, one.dtype);
    assert.equal(got.checksum, one.checksum);
  } finally { done(); }
});

test('add each class from typed arrays and read it back', { skip }, async () => {
  const { nxr, NxrError, demo: D, dir, done } = await demo();
  try {
    const read = (cls, name, o = {}) => nxr.read[cls]({ dataset: D, subject: 'S1', name, ...o });
    const add = (cls, o) => nxr.add[cls]({ dataset: D, subject: 'S1', ...o });

    const w = new Float32Array([0.5, -1.25, 3.0]);
    assert.equal((await add('field', { name: 'w', manifold: 'sensors', values: w, unit: 'mm', description: 'a test map' })).class, 'field');
    const back = await read('field', 'w');
    assert.equal(back.arrays.values.dtype, 'float32');
    same(back.arrays.values, w);                                                    // precision follows the input
    assert.match(JSON.stringify(back.producer), /values\.npy/);                    // the input's name, never a path
    await assert.rejects(add('field', { name: 'w', manifold: 'sensors', values: w }), (e) => e instanceof NxrError && e.code === 'already_exists');
    assert.ok((await add('field', { name: 'w', manifold: 'sensors', values: w, replace: true })).replaced);   // moved aside, never deleted
    await assert.rejects(add('field', { name: 'w4', manifold: 'sensors', values: new Float64Array([0, 1, 2, 3]) }), (e) => e.code === 'invalid_argument' && /is \[4\]/.test(e.message));
    await assert.rejects(add('field', { name: 'w5', manifold: 'sensors', values: { data: w, shape: [4] } }), (e) => e.code === 'invalid_argument' && /shape \[4\]/.test(e.message));   // the face checks shape against data

    await add('line', { name: 'epoch_time', n: 50, spacing: 0.01, unit: 's' });
    await add('line', { name: 'freq', n: 4, spacing: 1.0, origin: 1.0, unit: 'Hz', dimension: 'frequency' });
    const series = Float64Array.from({ length: CHANNELS * 50 }, (_, i) => 0.25 * i);
    await add('field', { name: 'series', manifold: 'sensors', time: 'epoch_time', values: { data: series, shape: [CHANNELS, 50] } });
    same((await read('field', 'series', { all: true })).arrays.values, series);

    const parts = new Int32Array([0, 1, 0]);
    assert.equal((await add('selection', { name: 'parts', manifold: 'sensors', type: 'partition', members: parts, entries: ['left', 'right'] })).members, 2);
    same((await read('selection', 'parts')).arrays.values, parts);
    await add('selection', { name: 'marks', manifold: 'epoch_time', type: 'spans', spans: { data: new Float64Array([0.05, 0.10, 0, 0.2, 0.35, 1]), shape: [2, 3] } });
    const s = await read('selection', 'marks');
    assert.deepEqual(Array.from(s.arrays.start), [0.05, 0.2]);
    assert.deepEqual(Array.from(s.arrays.code, Number), [0, 1]);
    await add('selection', { name: 'pick', manifold: 'sensors', type: 'indices', indices: new Int32Array([2, 0]) });

    await add('measurements', { selection: 'parts', field: 'w', measure: { mean: new Float64Array([1.5, -2.0]), max: new Float64Array([3.0, -1.25]) } });
    assert.deepEqual(Array.from((await read('selection', 'parts', { measure: 'mean', field: 'w' })).arrays.value), [1.5, -2.0]);
    let q = await nxr.query.measurements({ dataset: D, subject: 'S1', selection: 'parts', measure: 'max', field: 'w', where: 'io' });   // a stored ingredient, read as stored
    assert.deepEqual(q.result.rows.map((r) => r.value), [3.0, -1.25]);
    q = await nxr.query.measurements({ dataset: D, subject: 'S1', selection: 'parts', measure: 'max', field: 'w' });                      // the default, in memory: the same values
    assert.deepEqual(q.result.values.data, [3.0, -1.25]);

    const v = Float32Array.from({ length: 12 }, (_, i) => 0.5 * i);
    await add('statistic', { name: 'psd', measure: 'power_density', method: 'welch', axes: ['channel:sensors', 'frequency:freq'], of_field: 'rest', values: { data: v, shape: [CHANNELS, 4] } });
    const st = await read('statistic', 'psd');
    assert.equal(st.arrays.values.dtype, 'float32');
    same(st.arrays.values, v);

    const dense = new Float64Array([1, 0, 0, 0, 2, 0, 0.5, 0, 3]);
    await add('operator', { name: 'mix', from_manifold: 'sensors', to_manifold: 'sensors', values: { data: dense, shape: [3, 3] } });
    same((await read('operator', 'mix')).arrays.values, dense);
    await add('operator', { name: 'sparse', from_manifold: 'sensors', to_manifold: 'sensors', data: new Float64Array([4, 5]), indices: new Int32Array([0, 2]), indptr: new Int32Array([0, 1, 1, 2]) });
    const sp = await read('operator', 'sparse', { out: join(dir, 'sparse.npy') });   // `out` writes the .npy files as the command does, and says where
    assert.deepEqual(Array.from(sp.arrays.data), [4, 5]);
    assert.deepEqual(Array.from(sp.arrays.indices), [0, 2]);
    assert.deepEqual(Object.keys(sp.files).sort(), ['data', 'indices', 'indptr']);
    assert.ok(Object.values(sp.files).every((p) => existsSync(p)));

    const pos = new Float64Array([0, 0, 1, 0, 1, 0, 1, 0, 0, 0.5, 0.5, 0.5]);
    await add('points', { name: 'electrodes', positions: { data: pos, shape: [4, 3] } });
    same((await read('manifold', 'electrodes')).arrays.positions, pos);
    const faces = new Int32Array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
    assert.equal((await add('surface', { name: 'tetra', vertices: { data: pos, shape: [4, 3] }, faces: { data: faces, shape: [4, 3] }, no_primary: true })).n_vertices, 4);
    assert.deepEqual((await read('manifold', 'tetra')).arrays.cells_2.shape, [4, 3]);
    const names = new Set((await nxr.list.results({ dataset: D, subject: 'S1' })).results.map((r) => r.name));
    assert.ok(names.has('w') && names.has('psd'));
  } finally { done(); }
});

test('compile and run with from and to', { skip }, async () => {
  const { nxr, demo: D, done } = await demo();
  try {
    const c = await nxr.compile('psd', { dataset: D, subject: 'S1', set: { recording: 'rest' } });
    assert.equal(c.types, 'stored');
    assert.ok(c.plans);
    const params = { recording: 'rest', segment_seconds: 2.0, from: 0, to: 20 };
    const r = await nxr.run('psd', { dataset: D, subject: 'S1', ...params });
    assert.equal(r.cache, 'computed');
    assert.deepEqual(r.stored.map((s) => s.name), ['rest_psd_welch']);
    assert.equal((await nxr.run('psd', { dataset: D, subject: 'S1', ...params })).cache, 'hit');
    const psd = await nxr.read.statistic({ dataset: D, subject: 'S1', name: 'rest_psd_welch' });
    assert.equal(psd.arrays.values.shape[0], CHANNELS);
    assert.equal(psd.span.stop_sample, 2048);
    await assert.rejects(nxr.run('psd', { dataset: D, subject: 'S1', recording: 'rest' }), (e) => e.code === 'invalid_argument' && /time range/.test(e.message));
    const dry = await nxr.run('psd', { dataset: D, subject: 'S1', recording: 'rest', all: true, dry_run: true });
    assert.ok(dry.runs.every((run) => run.cache === 'planned'));
    assert.ok((await nxr.list.pipelines()).some((p) => p.name === 'psd'));
  } finally { done(); }
});

test('a write handle holds the lock across processes; a read handle keeps its database open and sees what is stored meanwhile', { skip }, async () => {
  const { nxr, NxrError, demo: D, done } = await demo();
  try {
    const child = `import { nxrNeuro } from ${JSON.stringify(pathToFileURL(join(PACKAGE, 'index.mjs')).href)};
      const nxr = await nxrNeuro();
      try { await nxr.add.line({ dataset: ${JSON.stringify(D)}, subject: 'S1', name: process.argv[1], n: 3, spacing: 1, unit: 's' }); }
      catch (e) { process.exit(e.code === 'writer_lock_held' ? 3 : 4); }`;
    const otherProcess = (name) => spawnSync(process.execPath, ['--input-type=module', '-e', child, name], { encoding: 'utf8' }).status;

    const d = nxr.open(D, 'write');
    assert.equal(otherProcess('blocked'), 3);                                                          // refused while the handle holds the lock
    await d.add.line({ subject: 'S1', name: 'mine', n: 3, spacing: 1.0, unit: 's' });                  // the handle's own commands borrow it
    assert.equal((await d.read.manifold({ subject: 'S1', name: 'mine' })).class, 'manifold');
    assert.ok(await d.show.dataset());
    d.close();
    assert.equal(d.closed, true);
    d.close();                                                                                         // twice is not an error
    assert.equal(otherProcess('after'), 0);                                                            // released by close()
    await assert.rejects(d.show.dataset(), (e) => e instanceof NxrError && e.code === 'invalid_argument' && /closed/.test(e.message));
    const r = nxr.open(D);                                                                             // a read handle takes no lock
    assert.deepEqual((await r.read.field({ subject: 'S1', name: 'rest', from: 0, to: 1 })).arrays.values.shape, [CHANNELS, 100]);
    assert.equal(otherProcess('while_read'), 0);
    assert.equal((await r.read.manifold({ subject: 'S1', name: 'sensors' })).class, 'manifold');
    await nxr.add.line({ dataset: D, subject: 'S1', name: 'stored_between', n: 4, spacing: 1.0, unit: 's' });   // another call, no handle, while the handle's reader is open
    assert.equal((await r.read.manifold({ subject: 'S1', name: 'stored_between' })).class, 'manifold');
    assert.ok((await r.run('psd', { subject: 'S1', recording: 'rest', segment_seconds: 2.0, all: true, dry_run: true })).runs.length);
    r.close();
    await assert.rejects(r.read.manifold({ subject: 'S1', name: 'sensors' }), /closed/);
    assert.throws(() => nxr.open(D, 'append'), /mode is 'read' or 'write'/);
  } finally { done(); }
});
