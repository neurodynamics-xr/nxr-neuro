// Shared by the tests: the built package (the folder holding index.mjs, nxr_io.mjs and nxr_io.wasm: the build's bindings/wasm-io, or the staged package) from NXR_WASM_PACKAGE.
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';

export const PACKAGE = process.env.NXR_WASM_PACKAGE;
export const skip = PACKAGE ? false : 'NXR_WASM_PACKAGE (the built package folder) is not set: bindings/wasm-io/test/run.sh builds it and runs the tests';
export async function load() {
  const m = await import(pathToFileURL(join(PACKAGE, 'index.mjs')).href);
  return { ...m, nxr: await m.nxrNeuro() };
}
