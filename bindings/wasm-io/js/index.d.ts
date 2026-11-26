// nxr-neuro in WebAssembly: every command of the `nxr-neuro` command line as `nxr.<verb>.<noun>(options)` (docs/specs/2026-11-19-io-service.md, step 5).
// The per-command options and the verbs are generated from cli/command_table.cpp (commands.d.ts).
import type { Verbs } from './commands.js';
export type * from './commands.js';

/** An Error carrying the library's code: `invalid_argument`, `no_such_row`, `ambiguous`, `already_exists`, `not_found`, `writer_lock_held`, `not_writable`, `schema_version`, ... */
export declare class NxrError extends Error { readonly code: string; constructor(message: string, code: string); }

export type TypedArray = Float32Array | Float64Array | Int8Array | Int16Array | Int32Array | BigInt64Array | Uint8Array | Uint16Array | Uint32Array | BigUint64Array;
/** an array a command takes: a typed array (its precision is kept), `{ data, shape, dtype? }` (C order; `shape` defaults to `[data.length]`), or the path of a .npy file (a string) */
export type ArrayInput = TypedArray | { data: TypedArray; shape?: readonly number[]; dtype?: string };
/** an array a command returns: a typed array in the stored precision, with its `shape` and `dtype` (a 64-bit integer array is a BigInt64Array) */
export type NamedArray = TypedArray & { readonly shape: number[]; readonly dtype: string };

/** What a command resolves to: its result object (the command line's `"stage": "result"` line). `arrays` holds the typed arrays by name when the command returns arrays (where the
 *  command line writes an array to a file or prints its checksum); it replaces the result's own list of their descriptions (name, dtype, shape); with none returned it is that list, empty. */
export type Result = Record<string, any> & { arrays?: Record<string, NamedArray> | unknown[] };

export interface RunOptions {
  dataset?: string;
  subject?: string;
  /** the time range, in seconds on the recording's clock: `from` and `to`, or `all` (none is refused) */
  from?: number;
  to?: number;
  all?: boolean;
  replace?: boolean;
  dry_run?: boolean;
  preset?: string;
  /** a group definition: the members (names, in order); its results go into the dataset's template */
  subjects?: string[];
  progress?: (line: Record<string, unknown>) => void;
  /** every other key is a parameter of the pipeline (`nxr.list.pipelines()`) */
  [parameter: string]: unknown;
}

/** a dataset opened once: the verbs with `dataset` bound, its databases kept open between calls, mode 'write' holding the writer lock until `close()` */
export type Handle = {
  readonly dataset: string;
  readonly mode: 'read' | 'write';
  readonly closed: boolean;
  run(pipeline: string, options?: Omit<RunOptions, 'dataset'>): Promise<Result>;
  close(): void;
} & { [V in keyof Verbs]: Verbs[V] };

export type Nxr = Verbs & {
  /** `nxr-neuro run PIPELINE`: a pipeline on a subject (or a group) */
  run(pipeline: string, options: RunOptions): Promise<Result>;
  /** a handle on a dataset: the verbs with `dataset` bound; mode 'write' takes the writer lock now and holds it until `close()` */
  open(dataset: string, mode?: 'read' | 'write'): Handle;
  /** the geometry compute on a mesh in memory (bindings/wasm's API) */
  compute: any;
  /** the Emscripten module (the bindings' raw functions) */
  module: any;
  version: string;
};

/** The whole of nxr-neuro in one module instance. */
export declare function nxrNeuro(options?: Record<string, unknown>): Promise<Nxr>;
/** A small synthetic dataset for tests and demos in the datastore folder; returns its folder. */
export declare function syntheticDataset(nxr: Nxr, datastore: string, options?: { name?: string; subject?: string; recording?: string; channels?: number; sfreq?: number; seconds?: number }): string;
