// wasmClient — the Cortical Flow Studio's `DataClient` (nxr-cortical-flow/backend/src/client.ts), in process, backed by nxr-neuro's io in WebAssembly.
// Spec: docs/specs/2026-11-11-wasm-io.md. Step 1: the reads (status, bundle, views, rows, arrays) and the array writes (create, put, delete).
// Step 2 (in progress): subjects.put/delete, fields, selections, measurements, operators, manifolds, dictionaries, jobs, layout, imports (Brainstorm sources) done; the other row writes (selections, fields, operators, manifolds, dictionaries, measurements, jobs, layout, imports) next:
// until then they throw an Error with code 'unsupported'.
//
//   import { wasmClient } from '@nxr/neuro-io'
//   const c = await wasmClient('/path/to/datastore')
//   const bundle = await c.subjects.get('study/sub-1.nxr.zarr')
//   const xyz = await c.read('study/sub-1.nxr.zarr', 'source/cortex/positions', { as: 'float64' })
import createNxrIo from './nxr_io.mjs';
import { wrap as wrapCompute } from './compute.mjs';   // bindings/wasm/js/wrap.mjs: the geometry compute's public surface

const TYPED = {
  float32: Float32Array, float64: Float64Array, int8: Int8Array, int16: Int16Array, int32: Int32Array, int64: BigInt64Array,
  uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, uint64: BigUint64Array, bool: Uint8Array,
};

/** an Error carrying nxr-neuro's io code (`not_found`, `invalid_argument`, `unsupported`, …) */
export class NxrIoError extends Error {
  constructor(message, code) { super(message); this.name = 'NxrIoError'; this.code = code; }
}

function unwrap(r) {
  if (r && typeof r === 'object' && 'error' in r && 'code' in r) throw new NxrIoError(r.error, r.code);
  return r;
}
const json = (r) => JSON.parse(unwrap(r));
const notYet = (what) => async () => { throw new NxrIoError(`${what}: not yet in nxr-neuro's wasm client (step 2 of docs/specs/2026-11-11-wasm-io.md)`, 'unsupported'); };

function typedOf(dtype, bytes) {
  const T = TYPED[dtype];
  if (!T) throw new NxrIoError(`a ${dtype} array is not read here`, 'unsupported');
  return new T(bytes.buffer, bytes.byteOffset, bytes.byteLength / T.BYTES_PER_ELEMENT);
}
// the Studio's `cast` (backend/src/read.ts): a 64-bit integer becomes a number first; the same type is not copied
function cast(src, as) {
  if (!as) return src;
  if (src instanceof BigInt64Array || src instanceof BigUint64Array) src = Float64Array.from(src, (v) => Number(v));
  const T = TYPED[as];
  if (!T) throw new NxrIoError(`as: ${as} is not a type the client reads`, 'invalid_argument');
  return src.constructor === T ? src : new T(src);
}
const dtypeOf = (data) => {
  for (const [k, T] of Object.entries(TYPED)) if (data instanceof T && k !== 'bool') return k;
  throw new NxrIoError('put: data must be a typed array', 'invalid_argument');
};

/** A `DataClient` on the datastore folder `root` (as the host's file system sees it: the real disk in Node). */
/** `program`: what a dataset this client makes is stamped as built by (default 'nxr-neuro'; the Studio's service says 'cortical-flow'). */
export async function wasmClient(root, { module, program } = {}) {
  const M = module ?? await createNxrIo();
  const ds = new M.Datastore(root, program ?? 'nxr-neuro');
  return {
    base: `wasm:${root}`,
    /** release the datasets held for writing (their writer locks) */
    close() { ds.close(); ds.delete(); },
    headers: {},
    async status() { return json(ds.status()); },
    subjects: {
      async get(storePath) { return json(ds.bundle(storePath)); },
      async put(id, patch, by) { return json(ds.updateSubject(id, JSON.stringify(patch ?? {}), by ?? '')); },
      async delete(id) { return json(ds.removeSubject(id)); },
    },
    views: { async get(view, where = {}, limit) { return json(ds.view(view, JSON.stringify(where), limit ?? 0)); } },
    rows: { async get(table, id) { return json(ds.row(table, id)); } },
    selections: {
      async post(row, children, by) { return json(ds.selection('create', JSON.stringify(row), children === undefined ? '' : JSON.stringify(children), by ?? '')); },
      async put(row, children, by) { return json(ds.selection('update', JSON.stringify(row), children === undefined ? '' : JSON.stringify(children), by ?? '')); },
      async delete(id) { json(ds.selection('delete', id, '', '')); },
    },
    fields: {
      async post(row, by, children) { return json(ds.createField(JSON.stringify(row), by ?? '', children === undefined ? '' : JSON.stringify(children))); },
      async put(row, by) { return json(ds.fieldUpdate(JSON.stringify(row), by ?? '')); },
      async delete(id) { json(ds.fieldDelete(id)); },
    },
    operators: {
      async post(row, children, by) { return json(ds.operator('create', JSON.stringify(row), children === undefined ? '' : JSON.stringify(children), by ?? '')); },
      async put(row, by) { return json(ds.operator('update', JSON.stringify(row), '', by ?? '')); },
      async delete(id) { json(ds.operator('delete', id, '', '')); },
    },
    manifolds: {
      async post(row, children, by) { return json(ds.manifold('create', JSON.stringify(row), children === undefined ? '' : JSON.stringify(children), by ?? '')); },
      async put(row, by) { return json(ds.manifold('update', JSON.stringify(row), '', by ?? '')); },
      async delete(id) { json(ds.manifold('delete', id, '', '')); },
    },
    dictionaries: {
      async post(row, children, by) { return json(ds.dictionary('create', JSON.stringify(row), children === undefined ? '' : JSON.stringify(children), by ?? '')); },
      async put(row, children, by) { return json(ds.dictionary('update', JSON.stringify(row), children === undefined ? '' : JSON.stringify(children), by ?? '')); },
      async delete(id) { json(ds.dictionary('delete', id, '', '')); },
    },
    measurements: {
      async post(rows, by) { return json(ds.measurements('post', JSON.stringify(rows), by ?? '')); },
      /** a query's roll-up, or (a selection id, no query) what its members carry — as the data service's GET */
      async get(query, selectionId) {
        if (selectionId && !query) return json(ds.measurements('kinds', JSON.stringify(selectionId), ''));
        if (query) return json(ds.measurements('query', JSON.stringify(query), ''));
        throw new NxrIoError('measurements: a query or a selection id', 'invalid_argument');
      },
      async delete(where) { return json(ds.measurements('delete', JSON.stringify(where), '')); },
    },
    jobs: {
      async get(id, where) { return json(ds.job('get', id ? JSON.stringify(id) : JSON.stringify(where ?? {}), '')); },
      async post(row, by) { return json(ds.job('create', JSON.stringify(row), by ?? '')); },
      async put(row) { return json(ds.job('update', JSON.stringify(row), '')); },
    },
    /** a Brainstorm source into a dataset, as the Studio's importSubjects: `nxr-neuro convert …` run in this module (the command line's own code) */
    imports: { async post(req) { return json(ds.importSubjects(JSON.stringify(req))); } },
    /** `nxr-neuro <argv>` in process — the command line itself: { code, lines, stderr } */
    async cli(argv) { return json(ds.cli(JSON.stringify(argv))); },
    /** an environment variable of the module (the host's environment is not passed in) — e.g. NXR_IO_DETERMINISTIC for the parity tests */
    setEnv(name, value) { ds.setEnv(name, String(value)); },
    async layout(storePath, q) { return json(ds.layout(storePath, JSON.stringify(q))); },
    /** DATASET OPERATIONS — the Studio's desktop app does these outside its DataClient (main's model service: ensureDataset, removeDataset, catalog) */
    datasets: {
      /** a dataset by name, made if absent (idempotent): its database, rows, folder and catalog entry; `source` {tool, protocol, path} */
      async create(name, source) { return json(ds.datasetCreate(name, source ? JSON.stringify(source) : '')); },
      /** the whole dataset — database, subject stores, catalog entry (D167); one not there is not an error */
      async delete(name) { return json(ds.datasetDelete(name)); },
      /** the catalog, rewritten from the databases (D140) */
      async catalog() { return json(ds.catalog()); },
    },
    /** the folder a store is in (an in-process client has no URL) */
    storeBase(storePath) { return `${root}/${storePath}`; },
    async read(storePath, arrayPath, opts = {}) {
      const r = unwrap(ds.read(storePath, arrayPath, opts.slice ? JSON.stringify(opts.slice) : ''));
      return cast(typedOf(r.dtype, r.bytes), opts.as);
    },
    arrays: {
      async create(storePath, arrayPath, meta) { return json(ds.create(storePath, arrayPath, JSON.stringify(meta))); },
      async put(storePath, arrayPath, data, region) {
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        unwrap(ds.put(storePath, arrayPath, bytes, dtypeOf(data), JSON.stringify(region)));
      },
      async delete(storePath, arrayPath) { unwrap(ds.remove(storePath, arrayPath)); },
    },
  };
}

/**
 * THE WHOLE OF nxr-neuro in one module instance: the geometry compute on meshes in memory (`compute`: the Manifold context — operators, eigenmodes, frames,
 * geodesics, transport, … — the API of bindings/wasm's standalone build) and data clients on datastores (`client(root)`: wasmClient, with every analysis verb
 * through `cli`). One instance, one heap: load it once (in a worker for long work).
 *
 *   const nxr = await nxrNeuro()
 *   const m = nxr.compute.manifold(vertices, faces)          // geometry in memory
 *   const c = await nxr.client('/path/to/datastore')         // a datastore: reads, writes, imports, analyses
 */
export async function nxrNeuro(options = {}) {
  const M = await createNxrIo(options);
  return { module: M, compute: wrapCompute(M), client: (root, o = {}) => wasmClient(root, { ...o, module: M }) };
}
