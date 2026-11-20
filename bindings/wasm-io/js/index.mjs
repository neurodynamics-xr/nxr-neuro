// nxr-neuro in WebAssembly, the JavaScript face (docs/specs/2026-11-19-io-service.md, step 5): every command of the `nxr-neuro` command line as a function,
// `nxr-neuro <verb> <noun> --option VALUE` is `nxr.<verb>.<noun>({ option: VALUE })`, a Promise. The module calls the same C++ library as the command line and the Python face
// (src/binding.cpp converts, it decides nothing); a dataset is a path (Node: NODERAWFS gives the module the real disk).
//
//   import { nxrNeuro } from '@nxr/neuro'
//   const nxr = await nxrNeuro()
//   await nxr.create.dataset({ dataset: 'study/demo' }); await nxr.create.subject({ dataset: 'study/demo', name: 'sub-01' })
//   await nxr.add.field({ dataset: 'study/demo', subject: 'sub-01', name: 'w', manifold: 'sensors', values: new Float32Array([0.5, -1.25, 3]) })
//   const r = await nxr.read.field({ dataset: 'study/demo', subject: 'sub-01', name: 'w' })      // r.arrays.values: a Float32Array with .shape and .dtype
//   const d = nxr.open('study/demo', 'write')                                                   // a handle: the dataset bound, its databases kept open, the writer lock held
//   await d.read.field({ subject: 'sub-01', name: 'w' }); d.close()
//
// The options are the command line's, with the dashes as underscores (`--from-sphere` is `from_sphere`; `--from` is `from`, `import` is `nxr.import`); a positional argument is
// its own key (`protocol`, `folder`, `file`) or the first argument. A flag is a boolean, a repeatable option an array. What a command prints as its result is the object it
// resolves to; where the command line writes an array to a file or prints its checksum, the object has `arrays`: typed arrays by name, each with `shape` and `dtype`.
// Arrays go in as typed arrays (or { data, shape, dtype }), or a path to a .npy file. A refusal is an NxrError with the library's `.code` (`invalid_argument`, `no_such_row`,
// `already_exists`, `writer_lock_held`, ...). `progress` (a function) receives each progress line as an object.
import { readFileSync } from 'node:fs';
import createNxrIo from './nxr_io.mjs';
import { wrap as wrapCompute } from './compute.mjs';   // bindings/wasm/js/wrap.mjs: the geometry compute's public surface
import { COMMANDS } from './commands.mjs';              // generated from cli/command_table.cpp (tools/command_table.mjs)
import { ARRAY_OPTIONS, JSON_OPTIONS } from './arrays.mjs';

/** An Error carrying the library's code (`invalid_argument`, `no_such_row`, `already_exists`, `writer_lock_held`, `not_writable`, `schema_version`, ...) */
export class NxrError extends Error {
  constructor(message, code) { super(message); this.name = 'NxrError'; this.code = code; }
}

const TYPED = {
  float32: Float32Array, float64: Float64Array, int8: Int8Array, int16: Int16Array, int32: Int32Array, int64: BigInt64Array,
  uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, uint64: BigUint64Array, bool: Uint8Array,
};
const DTYPE_OF = new Map([[Float32Array, 'float32'], [Float64Array, 'float64'], [Int8Array, 'int8'], [Int16Array, 'int16'], [Int32Array, 'int32'], [BigInt64Array, 'int64'],
  [Uint8Array, 'uint8'], [Uint16Array, 'uint16'], [Uint32Array, 'uint32'], [BigUint64Array, 'uint64']]);

const invalid = (message) => new NxrError(message, 'invalid_argument');
function unwrap(r) {
  if (r && typeof r === 'object' && 'error' in r && 'code' in r) throw new NxrError(r.error, r.code);
  return r;
}

/** an array the library handed back, as a typed array with `shape` and `dtype` */
function typedOf(a) {
  const T = TYPED[a.dtype];
  if (!T) throw new NxrError(`an array of dtype ${a.dtype} is not returned as a typed array`, 'unsupported');
  const t = new T(a.bytes.buffer, a.bytes.byteOffset, a.bytes.byteLength / T.BYTES_PER_ELEMENT);
  Object.defineProperties(t, { shape: { value: a.shape, enumerable: false }, dtype: { value: a.dtype, enumerable: false } });
  return t;
}

/** an array option's value for the library: a path (string) stays in the options; a typed array (or { data, shape, dtype }) goes in memory */
function memoryArray(option, name, value) {
  if (typeof value === 'string') return value;
  let data = value, shape = value?.shape, dtype = value?.dtype;
  if (value && !ArrayBuffer.isView(value) && typeof value === 'object' && ArrayBuffer.isView(value.data)) ({ data, shape, dtype } = value);
  if (!ArrayBuffer.isView(data) || data instanceof DataView) throw invalid(`${option}: an array is a typed array (Float32Array, Float64Array, Int32Array, ...), { data, shape, dtype } or the path of a .npy file`);
  const own = DTYPE_OF.get(data.constructor);
  if (!own) throw invalid(`${option}: a ${data.constructor.name} is not an array a command takes`);
  if (dtype && dtype !== own && !(dtype === 'bool' && own === 'uint8')) throw invalid(`${option}: dtype ${dtype} does not match its ${data.constructor.name}`);
  shape = shape ? Array.from(shape, Number) : [data.length];
  if (shape.reduce((n, d) => n * d, 1) !== data.length) throw invalid(`${option}: shape [${shape}] is not its ${data.length} values`);
  return { option, name: `${name}.npy`, dtype: dtype ?? own, shape, bytes: new Uint8Array(data.buffer, data.byteOffset, data.byteLength) };
}

const jsonOf = (options) => JSON.stringify(options, (_k, v) => (ArrayBuffer.isView(v) && !(v instanceof DataView) ? Array.from(v, Number) : typeof v === 'bigint' ? Number(v) : v));

function parseOptions(text) {
  return text.split(' ').filter(Boolean).map((w) => ({ name: w.replace(/[*+!]+$/, ''), required: w.includes('*'), multi: w.includes('+'), flag: w.includes('!') }));
}

/** the library's answer to one command: its result object, with the arrays (typed, with shape and dtype) in `arrays` when it returned any */
function resultOf(r) {
  const result = JSON.parse(r.result);
  if (r.arrays?.length) result.arrays = Object.fromEntries(r.arrays.map((a) => [a.name, typedOf(a)]));
  return result;
}

function makeApi(M) {
  /** one command, with the options the caller gave (`handle`: the number of an open handle, 0 none) */
  function call(handle, verb, noun, given) {
    const path = noun ? `${verb} ${noun}` : verb;
    const arrayOptions = ARRAY_OPTIONS[path] ?? [];
    const options = {};
    const arrays = [];
    let progress;
    for (const [k, v] of Object.entries(given)) {
      if (v === undefined || v === null) continue;
      if (k === 'progress') { progress = (line) => v(JSON.parse(line)); continue; }
      if (arrayOptions.includes(k)) {
        const m = memoryArray(k, k, v);
        if (typeof m === 'string') options[k] = m; else arrays.push(m);
      } else if (path === 'add measurements' && k === 'measure' && !Array.isArray(v) && typeof v === 'object') {
        const files = [];
        for (const [name, a] of Object.entries(v)) {
          const m = memoryArray(`measure.${name}`, name, a);
          if (typeof m === 'string') files.push(`${name}=${m}`); else arrays.push(m);
        }
        if (files.length) options[k] = files;
      } else if (JSON_OPTIONS[path]?.includes(k) && typeof v === 'object') {
        options[k] = JSON.stringify(v);   // a JSON option given as an object or a list: its text (a string is the text, or for `entries` the path of a JSON file)
      } else if (verb === 'compile' && !noun && (k === 'signatures' || k === 'inputs') && typeof v === 'string') {
        options[k] = JSON.parse(readFileSync(v, 'utf8'));   // a JSON file, read here (the library takes the object)
      } else if (verb === 'compile' && !noun && k === 'set' && !Array.isArray(v) && typeof v === 'object') {
        options[k] = Object.entries(v).map(([name, x]) => `${name}=${JSON.stringify(x)}`);
      } else options[k] = v;
    }
    return resultOf(unwrap(M.command(handle, verb, noun, jsonOf(options), arrays, progress)));
  }

  /** the function of one command: `fn(options)`, or `fn(positional, options)` for a command with a positional argument. `bound`: options set by a handle. */
  function make(verb, noun, positional, handleOf, bound = {}) {
    const pos = positional.replace(/\?$/, '');
    const fn = async function command(a, b) {
      let given;
      if (pos && (typeof a === 'string' || a instanceof URL)) given = { ...(b ?? {}), [pos]: String(a) };
      else if (a === undefined || a === null || (typeof a === 'object' && !ArrayBuffer.isView(a))) given = { ...(a ?? {}) };
      else throw invalid(`nxr.${verb}${noun ? `.${noun.replaceAll('-', '_')}` : ''}: the options are an object of named options${pos ? ` (or ${pos} first)` : ''}`);
      return call(handleOf ? handleOf() : 0, verb, noun, { ...given, ...bound });
    };
    // what the function takes, as the command's entry says it (the coverage test reads it against cli/command_table.cpp)
    const options = parseOptions(COMMANDS.find((c) => c[0] === verb && c[1] === noun)[3]).filter((o) => !(o.name in bound));
    return Object.assign(fn, { command: `${verb} ${noun}`.trim(), positional: pos, optionalPositional: positional.endsWith('?'), options });
  }

  /** `nxr-neuro list pipelines`: what `run` takes, each with its parameters' meaning, default and unit */
  async function pipelines() { return JSON.parse(unwrap(M.pipelines()).result); }

  /** the verbs with their nouns (`nxr.create.dataset`), each function made by `make`; with `dataset`, the ones that take a dataset, bound to it */
  function verbs(handleOf, dataset) {
    const out = {};
    for (const [verb, noun, positional, optionsText] of COMMANDS) {
      let bound = {};
      if (dataset !== undefined) {
        const names = new Set(parseOptions(optionsText).map((o) => o.name));
        const key = names.has('dataset') ? 'dataset' : positional === 'folder' ? 'folder' : null;
        if (!key) continue;
        bound = { [key]: dataset };
      }
      if (!noun) { out[verb] = make(verb, '', positional, handleOf, bound); continue; }
      (out[verb] ??= {})[noun.replaceAll('-', '_')] = verb === 'list' && noun === 'pipelines' ? pipelines : make(verb, noun, positional, handleOf, bound);
    }
    return out;
  }

  /** `nxr-neuro run PIPELINE --dataset D --subject S (--from S --to S | --all) [--<parameter> VALUE ...]`: a pipeline on a subject. The time range is the caller's: `from` and `to`
   *  (seconds on the recording's clock), or `all: true`; none is refused. Every other key is a parameter of the pipeline (`nxr.list.pipelines()`), except `preset`, `replace`,
   *  `dry_run`, `subjects`, `into` and `progress`. */
  function runner(handleOf, dataset) {
    return async function run(pipeline, options = {}) {
      const { dataset: d = dataset, subject, from, to, all, replace, dry_run, preset, subjects, into, progress, ...parameters } = options;
      for (const [old, now] of [['start', 'from'], ['stop', 'to']])
        if (old in parameters) throw invalid(`nxr.run(..., { ${old} }) is retired: the time range is from and to (seconds on the recording's clock), or all: true; give ${now} instead of ${old}`);
      if (d === undefined) throw invalid('nxr.run needs the dataset: run(pipeline, { dataset, subject, from, to | all })');
      const group = subjects !== undefined || into !== undefined;
      if (!group && subject === undefined) throw invalid('nxr.run needs the subject: run(pipeline, { dataset, subject, from, to | all })');
      const args = { pipeline, dataset: String(d), subject, parameters, from, to, all: !!all, replace: !!replace, dry_run: !!dry_run, preset, subjects, into };
      const say = progress ? (line) => progress(JSON.parse(line)) : undefined;
      return JSON.parse(unwrap(M.run(handleOf ? handleOf() : 0, jsonOf(args), say)).result);
    };
  }

  /** A dataset opened once (`nxr.open(dataset, mode)`): its functions are the verbs' with `dataset` bound (`d.read.field({ subject, name })`, `d.run('psd', { subject, all: true })`).
   *  The library keeps the subject's database open between the handle's reading calls (read-only: it never migrates, recovers or locks). mode 'write' takes the dataset's writer
   *  lock now and holds it until `close()`: every command through the handle borrows it, and another process's write is refused (`writer_lock_held`) meanwhile. 'read' takes no lock. */
  function open(dataset, mode = 'read') {
    if (mode !== 'read' && mode !== 'write') throw invalid(`mode is 'read' or 'write' (got ${JSON.stringify(mode)})`);
    const path = String(dataset);
    const number = unwrap(M.openHandle(path, mode)).handle;
    let closed = false;
    const handleOf = () => {
      if (closed) throw new NxrError(`the handle of ${path} is closed${mode === 'write' ? ' (its writer lock released)' : ''}: open it again`, 'invalid_argument');
      return number;
    };
    const handle = Object.assign(verbs(handleOf, path), {
      dataset: path,
      mode,
      run: runner(handleOf, path),
      /** close the open databases and release the writer lock; the handle's calls are refused afterwards (twice is not an error) */
      close() { if (!closed) { closed = true; unwrap(M.closeHandle(number)); } },
    });
    Object.defineProperty(handle, 'closed', { get: () => closed, enumerable: true });
    return handle;
  }

  return { ...verbs(), run: runner(), open, compute: wrapCompute(M), module: M, version: M.nxrVersion() };
}

/**
 * THE WHOLE OF nxr-neuro in one module instance: every command of the command line as `nxr.<verb>.<noun>(options)`, `nxr.run`, `nxr.open`, and the geometry compute on meshes in
 * memory (`nxr.compute`: the surface as data and the toolbox functions - operators, eigenmodes, frames, geodesics, transport, ...). One instance, one heap: load it once (in a worker
 * for long work).
 */
export async function nxrNeuro(options = {}) {
  return makeApi(await createNxrIo(options));
}

/** a small synthetic dataset for tests and demos (a recording of known signals with one bad span) in the datastore folder; returns its folder */
export function syntheticDataset(nxr, datastore, { name = 'demo', subject = 'S1', recording = 'rest', channels = 3, sfreq = 100, seconds = 40 } = {}) {
  return unwrap(nxr.module.syntheticDataset(datastore, name, subject, recording, channels, sfreq, seconds)).result;
}
