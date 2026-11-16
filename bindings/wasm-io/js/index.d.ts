// wasmClient — the Cortical Flow Studio's DataClient (nxr-cortical-flow/backend/src/client.ts) backed by nxr-neuro's io in WebAssembly.
type Row = Record<string, unknown>;
type As = 'float32' | 'float64' | 'int32' | 'uint32' | 'uint8';
export declare class NxrIoError extends Error { readonly code: string; }
export interface Catalog { schema_version: number; id: string; datasets: { id: string; name: string; path: string; database: string; subjects: number }[] }
export interface WasmDataClient {
  readonly base: string;
  /** release the datasets held for writing (their writer locks) */
  close(): void;
  readonly headers: Readonly<Record<string, string>>;
  status(): Promise<{ root: string; file: string; catalog: Catalog } | null>;
  subjects: { get(storePath: string): Promise<Row | null>; put(id: string, patch: Row, by?: string): Promise<Row>; delete(id: string): Promise<{ id: string }> };
  views: { get(view: string, where?: Record<string, string | number | null>, limit?: number): Promise<Row[]> };
  rows: { get(table: string, id: string): Promise<Row | null> };
  selections: { post(row: unknown, children?: unknown, by?: string): Promise<Row>; put(row: unknown, children?: unknown, by?: string): Promise<Row>; delete(id: string): Promise<void> };
  fields: { post(row: unknown, by?: string, children?: unknown): Promise<Row>; put(row: unknown, by?: string): Promise<Row>; delete(id: string): Promise<void> };
  operators: { post(row: unknown, children?: unknown, by?: string): Promise<Row>; put(row: unknown, by?: string): Promise<Row>; delete(id: string): Promise<void> };
  manifolds: { post(row: unknown, children?: unknown, by?: string): Promise<Row>; put(row: unknown, by?: string): Promise<Row>; delete(id: string): Promise<void> };
  dictionaries: { post(row: unknown, children?: unknown, by?: string): Promise<Row>; put(row: unknown, children?: unknown, by?: string): Promise<Row>; delete(id: string): Promise<void> };
  measurements: { post(rows: unknown, by?: string): Promise<number>; get(query?: unknown, selectionId?: string): Promise<Row[]>; delete(where: unknown): Promise<number> };
  jobs: { get(id?: string, where?: Row): Promise<Row | Row[] | null>; post(row: unknown, by?: string): Promise<Row>; put(row: unknown): Promise<Row> };
  imports: { post(req: unknown): Promise<{ subject: Row; counts: Record<string, number> }[]> };
  layout(storePath: string, q: { name: string; manifold?: string; type?: string; role?: 'time' | 'spectral' | 'eigen'; operator?: string; manifoldName?: string; factors?: readonly string[] }): Promise<{ path: string; folder: string }>;
  /** dataset operations (outside the Studio's DataClient: its main process's ensureDataset / removeDataset / catalog) */
  datasets: {
    create(name: string, source?: { tool: string; protocol: string; path?: string }): Promise<Row>;
    delete(name: string): Promise<{ deleted: string | null }>;
    catalog(): Promise<Catalog>;
  };
  storeBase(storePath: string): string;
  read(storePath: string, arrayPath: string, opts?: { as?: As; slice?: [number, number][] }): Promise<ArrayLike<number>>;
  arrays: {
    create(storePath: string, arrayPath: string, meta: { shape: readonly number[]; dtype: 'float32' | 'float64' | 'int32'; chunks?: readonly number[]; fill_value?: number | 'NaN' }): Promise<{ path: string; fresh: boolean }>;
    put(storePath: string, arrayPath: string, data: Float32Array | Float64Array | Int32Array, region: readonly (readonly [number, number])[]): Promise<void>;
    delete(storePath: string, arrayPath: string): Promise<void>;
  };
}
/** A DataClient on the datastore folder `root`. Step 1: reads and array writes; the row writes throw NxrIoError('unsupported'). */
export declare function wasmClient(root: string, options?: { module?: unknown; program?: string }): Promise<WasmDataClient>;
/** The whole of nxr-neuro in one module instance: the geometry compute (bindings/wasm's API) and data clients on datastores. */
export declare function nxrNeuro(options?: Record<string, unknown>): Promise<{ module: unknown; compute: any; client(root: string, options?: { program?: string }): Promise<WasmDataClient> }>;
