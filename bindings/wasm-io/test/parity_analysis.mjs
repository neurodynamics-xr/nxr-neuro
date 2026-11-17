// The ANALYSIS commands in WebAssembly against the native command: each `nxr-neuro <verb> …` run natively on one copy of a synthetic dataset and through
// wasmClient.cli (the same command line, compiled into the module) on another, deterministic on both sides (NXR_IO_DETERMINISTIC, its own seed per command);
// the exit code, the result lines and every table and file compared after each.
//   node parity_analysis.mjs <wasm-io build folder> <datastore (with the session subject S01 imported)> <native nxr-neuro>
import assert from 'node:assert/strict';
import { cpSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { keysOf, readKey } from './store_keys.mjs';   // the store's KEYS, at format 51 (files) and 52 (rows of the subject's small_files.sqlite)

const [, , PKG, STORE, NATIVE] = process.argv;
const { wasmClient } = await import(path.join(PKG, 'index.mjs'));
const A = `${STORE}-an-native`, B = `${STORE}-an-wasm`;
cpSync(STORE, A, { recursive: true });
cpSync(STORE, B, { recursive: true });
const wasm = await wasmClient(B);
const readerA = await wasmClient(A), readerB = await wasmClient(B);   // one decoder reads both copies' arrays
const REC = 'sub-S01_ses-01_task-rest_run-01_meg';
let checks = 0;

const TIMING = new Set(['seconds', 'elapsed', 'elapsed_s', 'ms', 'wall_s']);
const TIME = /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$/;
// a scalar is compared to 12 significant digits (the last-place rounding of the two builds, below), JSON text (producer_json, params_json) as its parsed value
const asJson = (t) => { if (!/^[[{]/.test(t)) return undefined; try { return JSON.parse(t); } catch { return undefined; } };
let DIGITS = 12;   // per command (an iterative solve: its tolerance, below)
const deep = (v, root) => typeof v === 'number' ? (Number.isInteger(v) ? v : Number(v.toPrecision(DIGITS)))
  : typeof v === 'string' ? (TIME.test(v) ? '<T>' : asJson(v) !== undefined ? deep(asJson(v), root) : v.split(root).join('<ROOT>')) : Array.isArray(v) ? v.map((x) => deep(x, root))
  : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, TIMING.has(k) ? '<elapsed>' : deep(x, root)])) : v;
// how long a step took (a measurement of the run, not of the data)
const rowsOf = (root, sql) => { const db = new DatabaseSync(path.join(root, 'd', 'dataset.sqlite'), { readOnly: true }); try { return JSON.parse(JSON.stringify(db.prepare(sql).all())); } finally { db.close(); } };
const files = (root) => keysOf(root).filter((f) => !/dataset\.sqlite|catalog\.json|\.lock$/.test(f));
async function same(what) {
  for (const { name: t } of rowsOf(A, "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")) {
    const q = t === 'datastore' ? 'SELECT id, driver FROM datastore' : `SELECT * FROM "${t}"`;
    const norm = (root) => rowsOf(root, q).map((r) => JSON.stringify(deep(r, root))).sort();
    assert.deepEqual(norm(B), norm(A), `${what}: table ${t}`);
  }
  assert.deepEqual(files(B), files(A), `${what}: files`);
  // the arrays: equal bytes, or — a computed array — equal VALUES to the last places. The same C++ compiled twice need not round alike: the native
  // arm64 build fuses multiply-adds (FMA) and sums in NEON lanes, WebAssembly does neither, so sums of many terms differ in their last bits.
  const arrays = new Set();
  for (const f of files(A)) {
    const a = readKey(A, f), b = readKey(B, f);
    if (f.endsWith('zarr.json')) { assert.deepEqual(deep(JSON.parse(b), B), deep(JSON.parse(a), A), `${what}: ${f}`); continue; }
    if (a.equals(b)) continue;
    const m = /^(d\/[^/]+\.nxr\.zarr)\/(.+?)\/c(\/|$)/.exec(f);
    assert.ok(m, `${what}: ${f} differs and is not an array chunk`);
    arrays.add(`${m[1]}\n${m[2]}`);
  }
  for (const key of arrays) {
    const [store, node] = key.split('\n');
    const va = await readerA.read(store, node), vb = await readerB.read(store, node);
    assert.equal(va.length, vb.length, `${what}: ${node} length`);
    const f64 = va instanceof Float64Array, tol = ARRAY_TOL ?? (f64 ? 1e-12 : 1e-6);
    let scale = 0, worst = 0;
    for (let i = 0; i < va.length; i++) scale = Math.max(scale, Math.abs(Number(va[i])));
    for (let i = 0; i < va.length; i++) {
      const x = Number(va[i]), y = Number(vb[i]);
      if (Number.isNaN(x) && Number.isNaN(y)) continue;
      worst = Math.max(worst, Math.abs(x - y) / (scale || 1));
    }
    assert.ok(worst <= tol, `${what}: ${node} differs by ${worst.toExponential(2)} of its largest value (allowed ${tol})`);
    if (worst > 0 && !roundings.has(node)) roundings.set(node, `${what}: ${node} ${va.constructor.name.replace('Array', '')} max |Δ| / max|x| = ${worst.toExponential(2)}`);
  }
  checks++;
}
const roundings = new Map();   // an array's rounding, reported once (the command that wrote it)

let seed = 5000;
let ARRAY_TOL;
// tolerance: last-place rounding by default; an ITERATIVE solve (the eigensolver: Spectra, stopped at its residual tolerance ~1e-10) to that tolerance — where it
// stops depends on the iteration's path, which the two builds' rounding changes
async function both(argv, ok = true, { digits = 12, arrayTol } = {}) {
  DIGITS = digits; ARRAY_TOL = arrayTol;
  seed += 1000;
  const arg = (root) => argv.map((x) => x.replace('<DS>', path.join(root, 'd')));
  const n = spawnSync(NATIVE, arg(A), { encoding: 'utf8', env: { ...process.env, NXR_IO_DETERMINISTIC: String(seed) } });
  wasm.setEnv('NXR_IO_DETERMINISTIC', String(seed));
  const w = await wasm.cli(arg(B));
  const lines = (out) => out.split('\n').filter(Boolean).map((l) => { try { return JSON.parse(l); } catch { return l; } });
  const what = argv.slice(0, 2).join(' ');
  assert.equal(w.code, n.status, `${what}: exit code (native stdout: ${n.stdout.slice(-300)} ${n.stderr.slice(-300)}; wasm: ${JSON.stringify(w).slice(-300)})`);
  if (ok) assert.equal(n.status, 0, `${what}: ${n.stdout.slice(-400)}${n.stderr.slice(-400)}`);
  // the JSON lines (the results); text lines are logs (the eigensolver's timings), not compared
  const json = (ls) => ls.filter((l) => l && typeof l === 'object');
  assert.deepEqual(deep(json(w.lines), B), deep(json(lines(n.stdout)), A), `${what}: its lines`);
  checks++;
  await same(what);
  console.log(`  ${what}: equal`);
}

const S = ['--dataset', '<DS>', '--subject', 'S01'];
await both(['noisecov', 'compute', ...S, '--recording', REC]);
await both(['psd', 'compute', ...S, '--recording', REC]);
await both(['lowpass', ...S, '--recording', REC, '--fpass', '20']);
await both(['headmodel', 'compute', ...S, '--meg', 'none', '--eeg-surface', 'cortex_pial_low']);   // the synthetic subject has no scalp: the sphere fitted to its cortex
await both(['inverse', 'compute', ...S, '--recording', REC]);
await both(['sourcepsd', 'compute', ...S, '--recording', REC, '--bands', 'alpha:8-12']);
await both(['tiles', 'compute', '--dataset', '<DS>', '--subject', 'sub-1']);
await both(['frames', 'compute', '--dataset', '<DS>', '--subject', '@default_subject', '--convention', 'ras', '--replace']);
// LAST: its rows stay in the dataset at the solver's tolerance, which a later command's comparison would hold to the last place
// a bumpy, asymmetric mesh of one piece (make_store.py: an icosphere with random radii): a symmetric one (the octahedron) has repeated eigenvalues, met only to the solver's tolerance, whose eigenvectors may turn freely within their space
await both(['eigen', 'solve', '--dataset', '<DS>', '--subject', 'sub-1', '--surface', 'bumpy', '--modes', '6'], true, { digits: 8, arrayTol: 1e-6 });
for (const r of roundings.values()) console.log(`  not bit-equal (within tolerance): ${r}`);
wasm.close();
console.log(`wasm-io analysis vs the native command: ${checks} checks passed`);
