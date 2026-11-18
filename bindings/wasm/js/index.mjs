// nxr-compute — WebAssembly compute backend for cortical surface analysis.
//
// Public ES-module API. Wraps the Emscripten-generated factory
// (createNxrComputeModule) so consumers don't need to know about WASM
// initialisation. Re-exports a friendly `initNxrCompute` that returns
// the loaded module.
//
// The functions are the library's, by toolbox name, on arrays; the surface is data (nothing is kept between calls):
//
//   1. Flat surface:
//        const nxr = await initNxrCompute()
//        const surface = nxr['geometry.surfaceArrays'](verts, faces)
//        const d = nxr['geometry.heat'](surface, [0])
//
//   2. Six-group nested namespace `nxr.manifold.*` over the surface data:
//        const mctx = nxr.createManifoldContext(verts, faces)
//        mctx.solve.eigen(300)
//        mctx.operator.d0()
//        mctx.measure.distance([0])
//        mctx.measure.distance.signed([0, 1, 2], true)
//
//      Or in functional form:
//        nxr.nxr.manifold.solve.eigen(mctx, 300)
//        nxr.nxr.manifold.operator.d0(mctx)
//
// An old name (`createContext`, `solve`, `heat`, ...) is refused with an error naming the new one.
//
// Repo / package: `@neurodynamics-xr/nxr-compute` (the engine).
// JS API namespace: `nxr.manifold.*` (the manifold concepts inside the engine).
// WASM artifact: `nxr_compute.wasm` (unchanged).

// dist/wasm/ holds the committed prebuilt artifacts (refreshed by
// scripts/build-wasm.sh from build_wasm/ after a successful build).
// index.mjs lives two levels below bindings/ (bindings/wasm/js/), so the
// artifact is `../../dist/wasm/`.
import createNxrComputeModule from '../../dist/wasm/nxr_compute.js'
import { wrap } from './wrap.mjs'

let _modulePromise = null

/**
 * Load the nxr-compute WASM module. Idempotent — subsequent calls return the
 * same module instance.
 *
 * @param {object} [options]
 * @param {(filename: string) => string} [options.locateFile] —
 *   Custom resolver for nxr_compute.wasm. By default the loader looks for
 *   nxr_compute.wasm alongside nxr_compute.js. Use this to override the location
 *   (e.g. when serving WASM from a CDN).
 * @returns {Promise<NxrCompute>}
 */
export async function initNxrCompute(options = {}) {
  if (_modulePromise) return _modulePromise
  _modulePromise = createNxrComputeModule(options).then(wrap)
  return _modulePromise
}


export { wrap }
export default initNxrCompute
